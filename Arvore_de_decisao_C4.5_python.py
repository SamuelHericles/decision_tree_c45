#!/usr/bin/env python
# coding: utf-8

# # Trabalho de reconhecimento de padrões
# + Implementar o algoritmo C4.5 de Árvore de Decisão para classificar o banco de dados fornecido
# + No algoritmo C4.5, você deve usar a entropia (ganho de informação) como critério de escolha dos nós
# + Não é necessário realizar as podas
# + Usar validação cruzada K-fold com K=10
# + Você deve escolher os atributos que achar mais convenientes
# + Deve-se usar pelo menos 4 atributos.

# # Imports necessários


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import warnings
warnings.filterwarnings('ignore')


# # Carregar base de dados
# 
# Criei um link no github para ficar mais facil carregar a base de dados, porque se não podia dificultar o meu programa procurar o arquivo nos diretórios de um computador. Além disso, tratei de nomear as colunas pois elas não estavam caracterizadas.

dados   = pd.read_csv('https://raw.githubusercontent.com/SamuelHericles/Arvore_decisao/master/wine.csv')
columns = {'Classe','Alcool','Acido malico','Cinza','Alcalinidade da cinza','Magnesio',
           'Fenois totais', 'Flavonoides','Fenois nao flavonoides','Proantocianidinas',
           'Intensidade da cor','Matiz','OD280/OD315 de vinhos diluidos','Prolina'}
dados.columns = columns
dados


# # Escolhas dos atributos
# Conforme dito no trabalho as colunas dadas já estão com os atributos a serem escolhidos, logo escolhi os quatro primeiros que são:
# + Prolina
# + Matiz
# + Magnesio
# + Cinza
# + Alcool
# 
# Não há problema em escolher outros atributos.

atributos = pd.DataFrame({})
atributos['Classe']  = dados['Classe'] 
atributos[dados.iloc[:,1].name]  = dados.iloc[:,1]
atributos[dados.iloc[:,2].name]  = dados.iloc[:,2]
atributos[dados.iloc[:,3].name]  = dados.iloc[:,3]
atributos[dados.iloc[:,4].name]  = dados.iloc[:,4]
atributos


# # Conforme dito, vamos remover uma classe
# Conforme aconselhado pelo professor alexandre, poderiamos remover uma classe da base para facilitar o trabalho, mas caso tivesse algunas dias a mais para a entrega acho que conseguia fazer para n classes.

atributos.drop(atributos.query('Classe==3').index,inplace=True)
atributos


# # Divide os dados de treino e teste com Kfold shuffle estratificado
# 
# Como havia comentado com o professor sobre o uso de kfold aleatório(shuffle) ele me recomendou para tratar da proporção entre as classes, então tratei assim:
# 
# + Na classe 1 como há 59 amostras pegei 10% delas que dá **6 amostras de teste** e **53 amostras de treinamento**
# + Na classe 2 como há 71 amostras pegei 10% delas que dá **7 amostras de teste** e **64 amostras de treinamento**
# 
# ao todo deu 13 amostras de teste escolhidas aleatóriamente e 117 amostras de teste.
# 
# A lógica da função abaixo é:
# 
# 1 - Divide a base entre as amostra da classe 1 e 2
# 
# 2 - Após divida pega 10% dos indices das amostras aleatóriamente
# 
# 3 - Os indices que não estão nos 10% aleatório vão para base de treinamento
# 
# 4 - Após feito os passos acima junta os indeces de trenio em um base chamada de **X** e teste chamda de **y** 
# 

# @param base  Base dados da situação
# return X     Base de treino
# return y     Base de teste 
def kfold_shuffle_estratificado(base):
    # Pegar a quantidade de dados que a classe 1 tem
    Classe1 = base.query('Classe==1')
    Classe1.reset_index(drop=True,inplace=True)

    # Pegar a quantidade de dados que a classe 2 tem
    Classe2 = base.query('Classe==2')
    Classe2.reset_index(drop=True,inplace=True)

    # Dividir os dados de treino e teste da classe 1
    index_teste_classe_1 = sorted(random.sample([i for i in Classe1.index.values],int(Classe1.shape[0]*.1)+1))
    X_classe_1 = Classe1.iloc[[i for i in Classe1.index if i not in index_teste_classe_1],:]
    y_classe_1 = Classe1.iloc[index_teste_classe_1,:]

    # Dividir os dados de treino e teste da classe 2
    index_teste_classe_2 = sorted(random.sample([i for i in Classe2.index.values],int(Classe2.shape[0]*.1)))
    X_classe_2 = Classe2.iloc[[i for i in Classe2.index if i not in index_teste_classe_2],:]
    y_classe_2 = Classe2.iloc[index_teste_classe_2,:]

    # Juntar dados de Treino e teste das classes
    X = X_classe_1
    y = y_classe_1

    X = X.append(X_classe_2,ignore_index=True)
    y = y.append(y_classe_2,ignore_index=True)
    return X,y


# # Entropia
# $H = - \sum_{i=1}^{n} {p_i(x)logp_i(x)}$
# # Ganho de informação
# $GH = H_{raiz} - \sum{pesos}*H_{folha}$
# 
# $Pesos = \frac{Nº amostras da folha}{Nº amostras da raiz}$
# 
# 
# As funções abaixo segue a equaçãoes da entropia e ganho acima, vale resaltar que usei a biblioteca math pois ela tem o recursos para alterar a base da função logaritmica pois com isso coloquei para base 2.

# @param base       Base de dados para entropia
# return entropia   Entropia da base de atributos
def calcula_entropia(base):
    # Pega quantidade de amostras da classe 1
    qt_am_1 = base.query('Classe==1').shape[0]
    
    # Pega quantidade de amostras da classe 2
    qt_am_2 = base.query('Classe==2').shape[0]
    
    # Pega o tamanho do vector de atributos
    qt_base = base.shape[0]
    
    # Calcula a probabilidade de cada classe 
    probabilidade_1  = qt_am_1/qt_base
    probabilidade_2  = qt_am_2/qt_base
    
    # Calcula a entropia
    Entropia = -1*(probabilidade_1*math.log(probabilidade_1,2) + probabilidade_2*math.log(probabilidade_2,2))
    return Entropia

# @param folha          Base de dados da folha
# @param entropia_pai   Entropia da base de dados completa
# @param entropia_filho Entropia do atributo especifíco
# return GH             Ganho de informação
def ganho_de_informacao(folha,entropia_pai,entropia_filho):
    
    # Pega os pelos de cada classe e coloca neste veto
    Pesos = [folha.query('Classe==1').shape[0]/folha.shape[0],folha.query('Classe==2').shape[0]/folha.shape[0]]
    
    # Calcula o ganho de informação do nó filho
    GH = (Pesos[0]*entropia_filho + Pesos[1]*entropia_filho) - entropia_pai
    return GH


# # Rotulagem da base dados pelo o limiar da mediana dos atributos
# 
# Essas duas funções são as mais importanes do algoritmo pois elas uma procura o melhor limiar baseado no ganho de informação da divisão dos rotulo do atributo a outra divide a base de dados do atributos para rotulá-la.
# + A função **caca_limiar** faça a caçada do melhor limiar que rotula melhor a base de dado, ela pega organiza a base de dados em ordem crescente e reseta os indices do vetor para pocura. Para realizar o algoritmo mais rápido, ele verifica se a classe de dois atributos são diferentes para efetuar a média dos dois  e assim verificar o ganho de informação com este. Por fim, quando pegar um vetor de todos os limiares e seus ganhos de informação, nos ordenamos em ordem descrecente com base no ganho de informação para obter o limiar com maior ganho e assim retornar este valor na função.
# + a função **divide_pelo_limiar** pega o limiar inserido na função após isso executa a função **ganho_de_informacao** após a rotulagem da base de dados, por fim retorna o ganho e informação o limiar escolhido.

# @param base            Base de dados de um atributo
# @param nome            Nome do atributo(feature)
# @param entropia_pai    Entropia da base de dado original
# return GH_best         Retorna o melhor ganho de informação do determinado limiar
# return limiar_best     Retorna o limiar do melhor ganho de informação
def caca_limiar(base,nome,entropia_pai):
    valores = base.sort_values(nome)
    valores.reset_index(drop=True,inplace=True)
    GH_limiar = []
    for i in range(valores.shape[0]-1):
        if valores.Classe[i] != valores.Classe[i+1]:
            limiar = 0
            limiar = (valores.iloc[i,1]+valores.iloc[i+1,1])/2
            GH_limiar.append(divide_pelo_limar(base.iloc[:,1],entropia_pai,limiar))
    GH_best,limiar_best = sorted(GH_limiar,reverse=True)[0]
    return GH_best,limiar_best

# @param base            Base de dados de um atributo
# @param entropia_pai    Entropia da base de dados da situação
# return GH              Ganho de informação do atributo
# return limiar          Limiar do atributo
def divide_pelo_limar(base,entropia_pai,limiar):
    
    # Cria um dataframe que pega a classe e o valor
    folha = pd.DataFrame(columns = {'Classe','Valor'})

    # Rotula os dados de teste a partir de cada limiar dos atributos
    for i in range(base.shape[0]):
         if base[i] > limiar:
                folha = folha.append({'Classe':1,'Valor':base[i]},
                                       ignore_index=True)
         elif base[i] <= limiar:
            folha = folha.append({'Classe':2,'Valor':base[i]},
                           ignore_index=True)

    # Após rotular os atributos calcula-se a entropia
    entropia_filho = calcula_entropia(folha)
    
    # Depois o ganho de informação
    GH = ganho_de_informacao(folha,entropia_pai,entropia_filho)
    
    return GH,limiar


# # O algoritmo de árvore de descisão C4.5
# 1. Para cada atributo:
# 
#     1.1 Ordene os atributos da base de treinamento do atributo específico;
#     
#     1.2 Determinar os limiares $\theta$;
#     
#     1.3 Para cada limiar $\theta$, determine as informações entre $\theta$ e os atributos.
#     
# 2. Escolha os pares [atributo,$\theta$] que oferece o mais alto ganho de informação.
# 
# 
# Fonte: slide disponivel pelo professor.
# 
# 
# + A função **rotula_rec** rotula recursivamente para criar as folhas da base de dados. No caso que foi escolhido 4 atributos é criado uma arvore com 4 de profundidade e $2^3 + 2^2 + 2 + 1$ = 15 folhas.
# + A função **arvore_de_decisao_c45** é a função principal que usa todas as funções acima.

# @param base       Dataframe do atributo específico
# @param GHs        Dataframe dos ganhos de informação e limiares
# @param i          Posição do dataframe
# return rotulos    Vetor dos rótulos atribuidos
def rotula_rec(base,GHs,i):
    rotulos = pd.DataFrame({})
    base.reset_index(drop=True,inplace=True)
    if i < GHs.shape[1]:
        for j in range(base.shape[0]):
            if base[GHs['Nome'][i]][j] >= GHs['Limiar'][i]:
                base['Classes_pred'][j] = 1
            else:
                base['Classes_pred'][j] = 2
    else:
        return
    rotulos = base.query('Classes_pred==1')
    rotulos = rotulos.append(base.query('Classes_pred==2'),ignore_index=True)
    
    rotula_rec(base.query('Classes_pred==1'),GHs,i+1)
    rotula_rec(base.query('Classes_pred==2'),GHs,i+1)
    
    return rotulos['Classes_pred']

# @param X       Base de treino
# @param y       Base de teste
# return         Acurácia do modelo
def arvore_de_decisao_c45(X,y):
    
    # Pega a entropia da base de dados
    entropia_pai = calcula_entropia(X)
    
    # Cria o dataframe que armazena os ganhos
    GHs = pd.DataFrame(columns = {'GH','Limiar','Nome'})
    
    # Pegar os dados de informação de cada atributo
    for coluna in X.columns[1:]:
        df_aux = pd.DataFrame({})
        df_aux = atributos[['Classe',coluna]]
        GH,limiar = caca_limiar(df_aux,coluna,entropia_pai)  
        GHs = GHs.append({'GH':GH,'Limiar':limiar,'Nome':coluna},ignore_index=True)

    # Organiza os ganhos de informação em ordem descrecente
    GHs.sort_values('GH',ascending=False,inplace=True)
    GHs.reset_index(drop=True,inplace=True)

    # Retira os rótulos da base de teste
    y_pred = y.iloc[:,1:]
    y_pred['Classes_pred'] = 0
    y_pred['Classes_pred'] = rotula_rec(y_pred,GHs,0)
    return (sum(y_pred['Classes_pred'] == y['Classe'])/y.shape[0])*100


# # Execução do algoritmo


# Vetor de acurácias
accs = []

# Cálcula várias vezes base de dados diferentes com árvore de descição
for _ in range(10):
    print(_)
    X,y = kfold_shuffle_estratificado(atributos)
    accs.append(arvore_de_decisao_c45(X,y))

# Plota cada resultado junto com a acurácia média do modelo
plt.title(f'Acurácia média do modelo é: {np.mean(accs).round(2)}%')
plt.plot(accs,'-')
plt.show()


# # Teste com sklearn


from sklearn import tree
from sklearn.model_selection import cross_validate
import graphviz
from sklearn.tree import export_graphviz
from sklearn.model_selection import GroupKFold

clf = tree.DecisionTreeClassifier()
results = cross_validate(clf, X.iloc[:,1:], X['Classe'], cv=10, return_train_score=False)
cv = GroupKFold(n_splits=10)
def imprime_resultados(results):
    
    media = results['test_score'].mean()*100
    print('Acurácia média %.2f' % media)
imprime_resultados(results)

features = X.columns[1:]
clf.fit(X.iloc[:,1:], X['Classe'])
dot_data = export_graphviz(clf, out_file = None, filled = True,
                rounded = True,
                feature_names = features)

