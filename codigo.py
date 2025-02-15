import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def filter_columns(df, filters: list): # adiciono no array o padrão que existe nas colunas e que não quero que tenha na saída final
    selected_columns = [True] * len(df.columns)  # Inicializa todas as colunas como True
    for index, column in enumerate(df.columns):
        if any(filter in column for filter in filters): selected_columns[index] = False
    return df[df.columns[selected_columns]]


def cleaning_dataset(df):
    _df = df.dropna(subset=df.columns.difference(['NOME']), how='all') # executa o dropna para todas as colunas sem visualizar a coluna NOME
    _df = _df[~_df.isna().all(axis=1)] # remove linhas com apenas NaN, se tiver algum dado na linha não remove
    return _df

def padronizar_colunas(df_concat):
    df_concat.columns = [re.sub(r'_\d{4}$', '', col) for col in df_concat.columns]
    return df_concat

if __name__ == '__main__':
    #Importando os dados
    df = pd.read_csv ('PEDE_PASSOS_DATASET_FIAP.csv',delimiter=';')

    # Hipotese -> Tivemos adição de colunas novas no dataset ao longo do tempo
    len(df.columns[df.columns.str.contains('2020')])
    df.columns[df.columns.str.contains('2020')]

    len(df.columns[df.columns.str.contains('2021')])
    df.columns[df.columns.str.contains('2021')]

    len(df.columns[df.columns.str.contains('2022')])
    df.columns[df.columns.str.contains('2022')]

    # Funções reutilizáveis

    def filter_columns(df, filters: list): # adiciono no array o padrão que existe nas colunas e que não quero que tenha na saída final
        selected_columns = [True] * len(df.columns)  # Inicializa todas as colunas como True
        for index, column in enumerate(df.columns):
            if any(filter in column for filter in filters): selected_columns[index] = False
        return df[df.columns[selected_columns]]


    def cleaning_dataset(df):
        _df = df.dropna(subset=df.columns.difference(['NOME']), how='all') # executa o dropna para todas as colunas sem visualizar a coluna NOME
        _df = _df[~_df.isna().all(axis=1)] # remove linhas com apenas NaN, se tiver algum dado na linha não remove
        return _df


    def plot_exact_counter(size, x, y, df) -> None:

        plt.figure(figsize=size)
        barplot = plt.bar(y.index, y.values)
        plt.xlabel(x)
        plt.ylabel('Count')

        for index, value in enumerate(y.values):
            plt.text(index, value, round(value, 2), color='black', ha="center")

        plt.show()


    def analyse_corr(df):
        df = df.apply(pd.to_numeric, errors='coerce')

        corr_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.show()

    # Separando os Data Frames por Ano
    df_2020 = filter_columns(df, ['2021', '2022'])
    len(df_2020) # com NaN

    df_2020 = cleaning_dataset(df_2020)
    len(df_2020)

    df_2020['FASE_TURMA_2020'].value_counts()

    df_2020.head()

    df_2021 = filter_columns(df, ['2020', '2022'])
    df_2021 = cleaning_dataset(df_2021)
    df_2021.head()

    df_2021['FASE_2021'].value_counts()

    df_2022 = filter_columns(df, ['2020', '2021'])
    df_2022 = cleaning_dataset(df_2022)
    df_2022.head()

    df_2022['FASE_2022'].value_counts()

    ##-----------------
    ### [Tratamento DF] Ajustes nos DF para junção dos 3.

    #Separa o campo Fase e Turma do ano de 2020 em 2

    # Usando regex para extrair números e letras
    df_2020['FASE_2020'] = df_2020['FASE_TURMA_2020'].str.extract(r'(\d+)')  # Extrai números
    df_2020['TURMA_2020'] = df_2020['FASE_TURMA_2020'].str.extract(r'(\D+)')  # Extrai letras

    df_2020['FASE_2020'].value_counts()

    #Remove registro da FASE = '207' O registro do aluna estava com informações inconsistentes.

    # Filtrando o registro onde a coluna "Fase" é igual a 207
    filtro = df_2020[df_2020['FASE_2020'] == '207']

    # Exibindo o resultado
    print(filtro)

    #Removerndo o Registro:
    df_2020 = df_2020.drop(df_2020[df_2020['FASE_2020'] == '207'].index)

    print('207' in df_2020['FASE_2020'].values)

    df_2020['FASE_2020'].value_counts()

    #Renomear as colunas de Recomendação do df 2022 para ficar igual a 2021 

    df_2022.rename(columns={'REC_AVA_1_2022': 'REC_EQUIPE_1_2022'}, inplace=True)
    df_2022.rename(columns={'REC_AVA_2_2022': 'REC_EQUIPE_2_2022'}, inplace=True)
    df_2022.rename(columns={'REC_AVA_3_2022': 'REC_EQUIPE_3_2022'}, inplace=True)
    df_2022.rename(columns={'REC_AVA_4_2022': 'REC_EQUIPE_4_2022'}, inplace=True)


    # Função para remover o sufixo do ano das colunas
    def padronizar_colunas(df_concat):
        df_concat.columns = [re.sub(r'_\d{4}$', '', col) for col in df_concat.columns]
        return df_concat

    # Padronizar os nomes das colunas
    df_2020 = padronizar_colunas(df_2020)
    df_2021 = padronizar_colunas(df_2021)
    df_2022 = padronizar_colunas(df_2022)

    # Adicionar a coluna do ano
    df_2020["ANO"] = 2020
    df_2021["ANO"] = 2021
    df_2022["ANO"] = 2022

    # Concatenar os DataFrames
    df_concat = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)

    # Removendo a coluna 'FASE_TURMA' diretamente no DataFrame original
    df_concat.drop('FASE_TURMA', axis=1, inplace=True)

    colunas = df_concat.columns
    print(colunas)

    # Verificando os tipos de dados das colunas
    tipos_de_dados = df_concat.dtypes

    print(tipos_de_dados)

    # Substituindo '#NULO!' por NaN

    df_concat.replace('#NULO!', np.nan, inplace=True)

    #Altera e Padroniza os campos de Indicadores 
    df_concat['IAA'] = df_concat['IAA'].astype(float).round(2)
    df_concat['IAN'] = df_concat['IAN'].astype(float).round(2)
    df_concat['IDA'] = df_concat['IDA'].astype(float).round(2)
    df_concat['IEG'] = df_concat['IEG'].astype(float).round(2)
    df_concat['INDE'] = df_concat['INDE'].astype(float).round(2)
    df_concat['IPP'] = df_concat['IPP'].astype(float).round(2)
    df_concat['IPS'] = df_concat['IPS'].astype(float).round(2)
    df_concat['IPV'] = df_concat['IPV'].astype(float).round(2)
    df_concat['FASE'] = df_concat['FASE'].astype(float).round(0)
    df_concat['FASE'] = df_concat['FASE'].astype(int)


    df_concat['FASE'].value_counts()
    df_concat['TURMA'].value_counts()

    ##-------------------------------------------------------------------------------------
    ## Eficácia das Intervenções: Iniciando as Análises

    df_concat.describe()

    ## ***Comparação Indicadores versus Fase e Ano***
    ### **Total Alunos por Fase e Turma**
    # Agrupar por FASE e ANO, contar o número de alunos
    total_alunos = df_concat.groupby(['FASE', 'ANO']).size().reset_index(name='Total_Alunos')

    # Criar uma nova coluna com os valores formatados para os rótulos
    total_alunos['TotalAlunos_Texto'] = total_alunos['Total_Alunos'].round(0).astype(str)

    # Converter a coluna 'ANO' para string (categorias)
    total_alunos['ANO'] = total_alunos['ANO'].astype(str)

    # Ordenar as fases em ordem crescente (0 a 10)
    total_alunos = total_alunos.sort_values(by='FASE')

    # Gráfico de barras agrupadas
    fig = px.bar(
        total_alunos,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='Total_Alunos',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Total de Alunos por Fase e Ano',
        text='TotalAlunos_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Total de Alunos',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range':[0,220]},
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='ANO',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()

    ##---------------------------------------------------------------------
    ### ***Comparativo de Indicadores por Fase e Ano - Preparativo DF***
    #cria o DF com os comparativos de Fase e Ano 

    # Agrupar por FASE e ANO, calcular a média dos indicadores
    comparacao_fase = df_concat.groupby(['FASE', 'ANO'])[['IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']].mean().reset_index()


    # Converter as colunas 'FASE' e 'ANO' para string (categorias)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)
    comparacao_fase['ANO'] = comparacao_fase['ANO'].astype(str)

    # Ordenar as fases em ordem crescente (0 a 10)
    comparacao_fase = comparacao_fase.sort_values(by='FASE', key=lambda x: x.astype(int))

    # Agrupar por FASE e ANO, calcular a média dos indicadores
    comparacao_fase_a = df_concat.groupby(['FASE', 'ANO'])[['IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']].mean().reset_index()

    # Calcular a média total do IAA por ano
    media_total_ano = df_concat.groupby('ANO')[['IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']].mean().reset_index()
    media_total_ano['FASE'] = 'Média Total'  # Adicionar uma coluna para identificar as médias totais

    # Concatenar os dados das fases com as médias totais
    comparacao_fase = pd.concat([comparacao_fase_a, media_total_ano], ignore_index=True)

    # Converter as colunas 'FASE' e 'ANO' para string (categorias)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)
    comparacao_fase['ANO'] = comparacao_fase['ANO'].astype(str)

    # Ordenar as fases em ordem crescente (0 a 10)
    comparacao_fase = comparacao_fase.sort_values(by=['FASE', 'ANO'])

    ##---------------------------------------------------------------------
    ### **IAA (Indicador de Auto Avaliação)**
    #IAA (Indicador de Auto Avaliação)

    # Garantir que a coluna 'FASE' seja tratada como string (categoria)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    comparacao_fase['IAA_Texto'] = comparacao_fase['IAA'].round(2).astype(str)

    # Gráfico de barras agrupadas
    fig = px.bar(
        comparacao_fase,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='IAA',   # Eixo Y: Valor médio do IAA
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Comparação do IAA (Indicador de Auto Avaliação) por Fase e Ano',
        text='IAA_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Média do IAA',
        yaxis={'range': [0, 11]},  # Definir o limite do eixo Y de 0 a 12
        #xaxis={'type': 'category'},  # Forçar o eixo X a ser tratado como categoria
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Forçar o eixo X a ser tratado como categoria e ordenar as fases
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda (garantir que os anos apareçam corretamente)
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
        #text=comparacao_fase['IAA'].round(2)  # Mostrar os valores nas barras
    )

    # Exibir o gráfico
    fig.show()

    ##--------------------------------------------------------------------------------------
    ### **IAN (Indicador de Adequação ao Nível)**
    #IAN (Indicador de Adequação ao Nível)

    # Garantir que a coluna 'FASE' seja tratada como string (categoria)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    comparacao_fase['IAN_Texto'] = comparacao_fase['IAN'].round(2).astype(str)

    # Gráfico de barras agrupadas
    fig = px.bar(
        comparacao_fase,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='IAN',   # Eixo Y: Valor médio do IAA
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Comparação do IAN (Indicador de Adequação ao Nível) por Fase e Ano',
        text='IAN_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Média do IAN',
        yaxis={'range': [0, 11]},  # Definir o limite do eixo Y de 0 a 12
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Forçar o eixo X a ser tratado como categoria e ordenar as fases
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda (garantir que os anos apareçam corretamente)
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delass
    )

    # Exibir o gráfico
    fig.show()

    ##---------------------------------------------------------------------
    ### **IDA (Indicador de Aprendizagem)**
    # Garantir que a coluna 'FASE' seja tratada como string (categoria)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    comparacao_fase['IDA_Texto'] = comparacao_fase['IDA'].round(2).astype(str)

    # Gráfico de barras agrupadas
    fig = px.bar(
        comparacao_fase,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='IDA',   # Eixo Y: Valor médio do IAA
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Comparação do IDA (Indicador de Aprendizagem) por Fase e Ano',
        text='IDA_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Média do IDA',
        yaxis={'range': [0, 11]},  # Definir o limite do eixo Y de 0 a 12
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Forçar o eixo X a ser tratado como categoria e ordenar as fases
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda (garantir que os anos apareçam corretamente)
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delass
    )

    # Exibir o gráfico
    fig.show()

    ##-------------------------------------------------
    ### **IEG (Indicador de Engajamento)**
    #IEG (Indicador de Engajamento)

    # Garantir que a coluna 'FASE' seja tratada como string (categoria)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    comparacao_fase['IEG_Texto'] = comparacao_fase['IEG'].round(2).astype(str)

    # Gráfico de barras agrupadas
    fig = px.bar(
        comparacao_fase,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='IEG',   # Eixo Y: Valor médio do IAA
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Comparação do IEG (Indicador de Engajamento) por Fase e Ano',
        text='IEG_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Média do IEG',
        yaxis={'range': [0, 11]},  # Definir o limite do eixo Y de 0 a 12
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Forçar o eixo X a ser tratado como categoria e ordenar as fases
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda (garantir que os anos apareçam corretamente)
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delass
    )

    # Exibir o gráfico
    fig.show()

    ##------------------------------------------------------------------------------------
    ### **INDE (Indice do Desenvolvimento Educacional)**
    #INDE (Indice do Desenvolvimento Educacional)

    # Garantir que a coluna 'FASE' seja tratada como string (categoria)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    comparacao_fase['INDE_Texto'] = comparacao_fase['INDE'].round(2).astype(str)

    # Gráfico de barras agrupadas
    fig = px.bar(
        comparacao_fase,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='INDE',   # Eixo Y: Valor médio do IAA
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Comparação do INDE (Indice do Desenvolvimento Educacional) por Fase e Ano',
        text='INDE_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Média do INDE',
        yaxis={'range': [0, 11]},  # Definir o limite do eixo Y de 0 a 12
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Forçar o eixo X a ser tratado como categoria e ordenar as fases
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda (garantir que os anos apareçam corretamente)
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delass
    )

    # Exibir o gráfico
    fig.show()

    ##----------------------------------------------------------------------------------
    ### **IPP (Indicador Psicopedagógico)**
    #IPP (Indicador Psicopedagógico)

    # Garantir que a coluna 'FASE' seja tratada como string (categoria)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    comparacao_fase['IPP_Texto'] = comparacao_fase['IPP'].round(2).astype(str)

    # Gráfico de barras agrupadas
    fig = px.bar(
        comparacao_fase,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='IPP',   # Eixo Y: Valor médio do IAA
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Comparação do IPP (Indicador Psicopedagógico) por Fase e Ano',
        text='IPP_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Média do IPP',
        yaxis={'range': [0, 11]},  # Definir o limite do eixo Y de 0 a 12
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Forçar o eixo X a ser tratado como categoria e ordenar as fases
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda (garantir que os anos apareçam corretamente)
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delass
    )

    # Exibir o gráfico
    fig.show()

    ##-------------------------------------------------------------------------------
    ### **IPS (Indicador Psicossocial)**
    #IPS (Indicador Psicossocial)

    # Garantir que a coluna 'FASE' seja tratada como string (categoria)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    comparacao_fase['IPS_Texto'] = comparacao_fase['IPS'].round(2).astype(str)

    # Gráfico de barras agrupadas
    fig = px.bar(
        comparacao_fase,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='IPS',   # Eixo Y: Valor médio do IAA
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Comparação do IPS (Indicador Psicossocial) por Fase e Ano',
        text='IPS_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Média do IPS',
        yaxis={'range': [0, 11]},  # Definir o limite do eixo Y de 0 a 12
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Forçar o eixo X a ser tratado como categoria e ordenar as fases
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda (garantir que os anos apareçam corretamente)
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delass
    )

    # Exibir o gráfico
    fig.show()

    ##--------------------------------------------------------------------------------
    ### **IPV (Indicador de Ponto de Virada)**
    # Garantir que a coluna 'FASE' seja tratada como string (categoria)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    comparacao_fase['IPV_Texto'] = comparacao_fase['IPV'].round(2).astype(str)

    # Gráfico de barras agrupadas
    fig = px.bar(
        comparacao_fase,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='IPV',   # Eixo Y: Valor médio do IAA
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Comparação do IPV (Indicador de Ponto de Virada) por Fase e Ano',
        text='IPV_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Média do IPV',
        yaxis={'range': [0, 11]},  # Definir o limite do eixo Y de 0 a 12
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Forçar o eixo X a ser tratado como categoria e ordenar as fases
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda (garantir que os anos apareçam corretamente)
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delass
    )

    # Exibir o gráfico
    fig.show()

    ##---------------------------------------------------------------------------------------
    ### ***Analise de Box Plot***
    # Lista de indicadores
    indicadores = ['IDA', 'IAA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']

    # Lista de anos disponíveis
    anos = df_concat['ANO'].unique()

    # Função para criar boxplots para um indicador
    def criar_boxplots_por_indicador(indicador):
        # Criar subplots com 1 linha e 3 colunas (um gráfico por ano)
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f'Box Plot - {indicador} por Fase - {ano}' for ano in anos])
        
        # Adicionar um boxplot para cada ano
        for i, ano in enumerate(anos):
            # Filtrar os dados para o ano atual
            df_filtrado = df_concat[df_concat['ANO'] == ano]
            
            # Criar o boxplot
            boxplot = go.Box(
                x=df_filtrado['FASE'],
                y=df_filtrado[indicador],
                name=str(ano)  # Converter o ano para string
            )
            
            # Adicionar o boxplot ao subplot correspondente
            fig.add_trace(boxplot, row=1, col=i+1)
        
        # Ajustar o layout
        fig.update_layout(
            title_text=f"Boxplots de {indicador} por Fase e Ano",
            showlegend=False,
            height=400,
            width=1200
        )
        
        # Exibir o gráfico
        fig.show()

    # Gerar boxplots para todos os indicadores
    for indicador in indicadores:
        criar_boxplots_por_indicador(indicador)

    ##----------------------------------------------------------------------------------------
    ## **Avaliação das Recomendações**
    ### ***Recomendações Equipe 1***
    #Equipe 1

    # Cria o Df para o Grafico:

    # Agrupar por REC_EQUIPE_1 e ANO, contar o número de alunos
    total_recomendacao_equipe1 = df_concat.groupby(['REC_EQUIPE_1', 'ANO']).size().reset_index(name='count')

    # Calcular o total de alunos por ano
    total_recomendacao_equipe1['Total_Alunos_Ano'] = total_recomendacao_equipe1.groupby('ANO')['count'].transform('sum')

    # Calcular a porcentagem de cada recomendação em relação ao total de alunos do ano
    total_recomendacao_equipe1['Represent'] = (total_recomendacao_equipe1['count'] / total_recomendacao_equipe1['Total_Alunos_Ano']) * 100

    #Cria o Grafico:

    # Converter a coluna 'ANO' para string (categorias)
    total_recomendacao_equipe1['ANO'] = total_recomendacao_equipe1['ANO'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    total_recomendacao_equipe1['Represent_Texto'] = total_recomendacao_equipe1['Represent'].astype(float).round(2).astype(str)+ '%'

    # Ordenar as fases em ordem crescente (0 a 10)
    total_recomendacao_equipe1 = total_recomendacao_equipe1.sort_values(by='REC_EQUIPE_1')

    # Gráfico de barras agrupadas
    fig = px.bar(
        total_recomendacao_equipe1,
        x='REC_EQUIPE_1',  # Eixo X: Fases (tratadas como categorias)
        y='Represent',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Representatividade de Alunos por Recomendação Equipe 1',
        text='Represent_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range': [0, 70]},  # Definir o limite do eixo Y 
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()

    ##------------------------------------------------------------------------------------------
    ### ***Recomendações Equipe 2***
    #Equipe 2

    # Cria o Df para o Grafico:

    # Agrupar por REC_EQUIPE_2 e ANO, contar o número de alunos
    total_recomendacao_equipe2 = df_concat.groupby(['REC_EQUIPE_2', 'ANO']).size().reset_index(name='count')

    # Calcular o total de alunos por ano
    total_recomendacao_equipe2['Total_Alunos_Ano'] = total_recomendacao_equipe2.groupby('ANO')['count'].transform('sum')

    # Calcular a porcentagem de cada recomendação em relação ao total de alunos do ano
    total_recomendacao_equipe2['Represent'] = (total_recomendacao_equipe2['count'] / total_recomendacao_equipe2['Total_Alunos_Ano']) * 100

    #Cria o Grafico:

    # Converter a coluna 'ANO' para string (categorias)
    total_recomendacao_equipe2['ANO'] = total_recomendacao_equipe2['ANO'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    total_recomendacao_equipe2['Represent_Texto'] = total_recomendacao_equipe2['Represent'].astype(float).round(2).astype(str)+ '%'

    # Ordenar as fases em ordem crescente (0 a 10)
    total_recomendacao_equipe2 = total_recomendacao_equipe2.sort_values(by='REC_EQUIPE_2')

    # Gráfico de barras agrupadas
    fig = px.bar(
        total_recomendacao_equipe2,
        x='REC_EQUIPE_2',  # Eixo X: Fases (tratadas como categorias)
        y='Represent',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Representatividade de Alunos por Recomendação Equipe 2',
        text='Represent_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range': [0, 70]},  # Definir o limite do eixo Y 
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()

    ##-------------------------------------------------------------------------------------------
    ### ***Recomendações Equipe 3***
    #Equipe 3

    # Cria o Df para o Grafico:

    # Agrupar por REC_EQUIPE_3 e ANO, contar o número de alunos
    total_recomendacao_equipe3 = df_concat.groupby(['REC_EQUIPE_3', 'ANO']).size().reset_index(name='count')

    # Calcular o total de alunos por ano
    total_recomendacao_equipe3['Total_Alunos_Ano'] = total_recomendacao_equipe3.groupby('ANO')['count'].transform('sum')

    # Calcular a porcentagem de cada recomendação em relação ao total de alunos do ano
    total_recomendacao_equipe3['Represent'] = (total_recomendacao_equipe3['count'] / total_recomendacao_equipe3['Total_Alunos_Ano']) * 100

    #Cria o Grafico:

    # Converter a coluna 'ANO' para string (categorias)
    total_recomendacao_equipe3['ANO'] = total_recomendacao_equipe3['ANO'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    total_recomendacao_equipe3['Represent_Texto'] = total_recomendacao_equipe3['Represent'].astype(float).round(2).astype(str)+ '%'

    # Ordenar as fases em ordem crescente (0 a 10)
    total_recomendacao_equipe3 = total_recomendacao_equipe3.sort_values(by='REC_EQUIPE_3')

    # Gráfico de barras agrupadas
    fig = px.bar(
        total_recomendacao_equipe3,
        x='REC_EQUIPE_3',  # Eixo X: Fases (tratadas como categorias)
        y='Represent',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Representatividade de Alunos por Recomendação Equipe 3',
        text='Represent_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range': [0, 70]},  # Definir o limite do eixo Y 
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()

    ##---------------------------------------------------------------------------------------
    ### ***Recomendações Equipe 4***
    #Equipe 4

    # Cria o Df para o Grafico:

    # Agrupar por REC_EQUIPE_4 e ANO, contar o número de alunos
    total_recomendacao_equipe4 = df_concat.groupby(['REC_EQUIPE_4', 'ANO']).size().reset_index(name='count')

    # Calcular o total de alunos por ano
    total_recomendacao_equipe4['Total_Alunos_Ano'] = total_recomendacao_equipe4.groupby('ANO')['count'].transform('sum')

    # Calcular a porcentagem de cada recomendação em relação ao total de alunos do ano
    total_recomendacao_equipe4['Represent'] = (total_recomendacao_equipe4['count'] / total_recomendacao_equipe4['Total_Alunos_Ano']) * 100

    #Cria o Grafico:

    # Converter a coluna 'ANO' para string (categorias)
    total_recomendacao_equipe4['ANO'] = total_recomendacao_equipe4['ANO'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    total_recomendacao_equipe4['Represent_Texto'] = total_recomendacao_equipe4['Represent'].astype(float).round(2).astype(str)+ '%'

    # Ordenar as fases em ordem crescente (0 a 10)
    total_recomendacao_equipe4 = total_recomendacao_equipe4.sort_values(by='REC_EQUIPE_4')

    # Gráfico de barras agrupadas
    fig = px.bar(
        total_recomendacao_equipe4,
        x='REC_EQUIPE_4',  # Eixo X: Fases (tratadas como categorias)
        y='Represent',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Representatividade de Alunos por Recomendação Equipe 4',
        text='Represent_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range': [0, 90]},  # Definir o limite do eixo Y 
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()

    ### ***Correlação de Variaveis - Alunos Com Recomendação de Promoção***
    # Filtrar apenas os alunos com a recomendação "Promovido de Fase"
    # Filtrar os alunos que foram recomendados como "Promovido de Fase" ou "Promovido de Fase + Bolsa" pela equipe 1 OU equipe 2
    df_promovido = df_concat[
        (df_concat['REC_EQUIPE_1'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])) |
        (df_concat['REC_EQUIPE_2'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])) |
        (df_concat['REC_EQUIPE_3'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])) |
        (df_concat['REC_EQUIPE_4'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])) 
    ]

    df_promovido.loc[:, 'Turma_'] = df_promovido['TURMA'].astype('category').cat.codes

    # Selecionar as colunas de interesse
    variaveis = ['IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV','FASE','Turma_']
    df_correlacao = df_promovido[variaveis]

    # Exibir o DataFrame com as variáveis selecionadas
    #print(df_correlacao)
    # Calcular a matriz de correlação
    matriz_correlacao = df_correlacao.corr()

    # Exibir a matriz de correlação
    #print(matriz_correlacao)

    # Criar o heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    # Ajustar o layout
    plt.title('Matriz de Correlação - Alunos Promovidos de Fase ou Promovidos de Fase + Bolsa (Equipe 1 ou Equipe 2)')
    plt.show()

    #---------------------------------------------------------------------------------------

    ### **Avaliação de Fases que possuem maior representatividade***
    # Agrupar por FASE e ANO, contar o número de alunos
    total_alunos = df_concat.groupby(['FASE', 'ANO']).size().reset_index(name='Total_Alunos')

    ### Representatividade de Alunos Promovidos por Fase e Ano - Equipe 1
    # Filtra promovidos Equipe
    df_filtered_eq1 = df_concat[df_concat['REC_EQUIPE_1'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])]

    # Agrupar por ano e contar o número de alunos promovidos
    df_grouped_eq1 = df_filtered_eq1.groupby(['ANO','FASE']).size().reset_index(name='count')

    # Junta a tabela de total de alunos por Fase e Ano par
    df_grouped_eq1 = pd.merge(df_grouped_eq1,total_alunos, on =['ANO','FASE'], how='left')

    # Calcular a porcentagem de alunos promovidos pelo total de alunos na fase e ano
    df_grouped_eq1['Represent'] = (df_grouped_eq1['count'] / df_grouped_eq1['Total_Alunos']) * 100

    #Cria o Grafico:

    # Converter a coluna 'ANO' para string (categorias)
    df_grouped_eq1['ANO'] = df_grouped_eq1['ANO'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    df_grouped_eq1['Represent_Texto'] = df_grouped_eq1['Represent'].astype(float).round(2).astype(str)+ '%'

    # Ordenar as fases em ordem crescente (0 a 10)
    df_grouped_eq1 = df_grouped_eq1.sort_values(by='FASE')

    # Gráfico de barras agrupadas
    fig = px.bar(
        df_grouped_eq1,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='Represent',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Representatividade de Alunos Promovidos pela Equipe 1',
        text='Represent_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range': [0, 110]},  # Definir o limite do eixo Y 
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()

    ##-------------------------------------------------------------------------------
    ### Representatividade de Alunos Promovidos por Fase e Ano - Equipe 2
    # Filtra promovidos Equipe
    df_filtered_eq2 = df_concat[df_concat['REC_EQUIPE_2'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])]

    # Agrupar por ano e contar o número de alunos promovidos
    df_grouped_eq2 = df_filtered_eq2.groupby(['ANO','FASE']).size().reset_index(name='count')

    # Junta a tabela de total de alunos por Fase e Ano par
    df_grouped_eq2 = pd.merge(df_grouped_eq2,total_alunos, on =['ANO','FASE'], how='left')

    # Calcular a porcentagem de alunos promovidos pelo total de alunos na fase e ano
    df_grouped_eq2['Represent'] = (df_grouped_eq2['count'] / df_grouped_eq2['Total_Alunos']) * 100

    #Cria o Grafico:

    # Converter a coluna 'ANO' para string (categorias)
    df_grouped_eq2['ANO'] = df_grouped_eq2['ANO'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    df_grouped_eq2['Represent_Texto'] = df_grouped_eq2['Represent'].astype(float).round(2).astype(str)+ '%'

    # Ordenar as fases em ordem crescente (0 a 10)
    df_grouped_eq2 = df_grouped_eq2.sort_values(by='FASE')

    # Gráfico de barras agrupadas
    fig = px.bar(
        df_grouped_eq2,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='Represent',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Representatividade de Alunos Promovidos pela Equipe 2',
        text='Represent_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range': [0, 110]},  # Definir o limite do eixo Y 
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()

    ##-------------------------------------------------------------------------------
    ### Representatividade de Alunos Promovidos por Fase e Ano - Equipe 3
    # Filtra promovidos Equipe
    df_filtered_eq3 = df_concat[df_concat['REC_EQUIPE_3'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])]

    # Agrupar por ano e contar o número de alunos promovidos
    df_grouped_eq3 = df_filtered_eq3.groupby(['ANO','FASE']).size().reset_index(name='count')

    # Junta a tabela de total de alunos por Fase e Ano par
    df_grouped_eq3 = pd.merge(df_grouped_eq3,total_alunos, on =['ANO','FASE'], how='left')

    # Calcular a porcentagem de alunos promovidos pelo total de alunos na fase e ano
    df_grouped_eq3['Represent'] = (df_grouped_eq3['count'] / df_grouped_eq3['Total_Alunos']) * 100

    #Cria o Grafico:

    # Converter a coluna 'ANO' para string (categorias)
    df_grouped_eq3['ANO'] = df_grouped_eq3['ANO'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    df_grouped_eq3['Represent_Texto'] = df_grouped_eq3['Represent'].astype(float).round(2).astype(str)+ '%'

    # Ordenar as fases em ordem crescente (0 a 10)
    df_grouped_eq3 = df_grouped_eq3.sort_values(by='FASE')

    # Gráfico de barras agrupadas
    fig = px.bar(
        df_grouped_eq3,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='Represent',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Representatividade de Alunos Promovidos pela Equipe 3',
        text='Represent_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range': [0, 110]},  # Definir o limite do eixo Y 
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()

    ##-------------------------------------------------------------------------------
    ### Representatividade de Alunos Promovidos por Fase e Ano - Equipe 4

    # Filtra promovidos Equipe
    df_filtered_eq4 = df_concat[df_concat['REC_EQUIPE_4'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])]

    # Agrupar por ano e contar o número de alunos promovidos
    df_grouped_eq4 = df_filtered_eq4.groupby(['ANO','FASE']).size().reset_index(name='count')

    # Junta a tabela de total de alunos por Fase e Ano par
    df_grouped_eq4 = pd.merge(df_grouped_eq4,total_alunos, on =['ANO','FASE'], how='left')

    # Calcular a porcentagem de alunos promovidos pelo total de alunos na fase e ano
    df_grouped_eq4['Represent'] = (df_grouped_eq4['count'] / df_grouped_eq4['Total_Alunos']) * 100

    #Cria o Grafico:

    # Converter a coluna 'ANO' para string (categorias)
    df_grouped_eq4['ANO'] = df_grouped_eq4['ANO'].astype(str)

    # Criar uma nova coluna com os valores formatados para os rótulos
    df_grouped_eq4['Represent_Texto'] = df_grouped_eq4['Represent'].astype(float).round(2).astype(str)+ '%'

    # Ordenar as fases em ordem crescente (0 a 10)
    df_grouped_eq4 = df_grouped_eq4.sort_values(by='FASE')

    # Gráfico de barras agrupadas
    fig = px.bar(
        df_grouped_eq4,
        x='FASE',  # Eixo X: Fases (tratadas como categorias)
        y='Represent',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Representatividade de Alunos Promovidos pela Equipe 4',
        text='Represent_Texto'  # Usar a coluna formatada como texto nas barras
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range': [0, 110]},  # Definir o limite do eixo Y 
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='Ano',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()
    ##-------------------------------------------------------------------------------

    ## **Análise da Distribuição de Alunos por Pedra-Conceito ao Longo dos Anos**

    # Agrupar por FASE e ANO, contar o número de alunos
    total_alunos = df_concat.groupby(['PEDRA', 'ANO']).size().reset_index(name='Total_Alunos')

    # Criar uma nova coluna com os valores formatados para os rótulos
    total_alunos['TotalAlunos_Texto'] = total_alunos['Total_Alunos'].round(0).astype(str)

    # Converter a coluna 'ANO' para string (categorias)
    total_alunos['ANO'] = total_alunos['ANO'].astype(str)

    # Ordenar as pedras em ordem crescente (0 a 10)
    total_alunos = total_alunos.sort_values(by='PEDRA')

    # Exibir o dataframe
    total_alunos

    # Gráfico de barras agrupadas
    fig = px.bar(
        total_alunos,
        x='PEDRA',  # Eixo X: Pedras
        y='Total_Alunos',   # Eixo Y: Total de alunos
        color='ANO',  # Cores diferentes para cada ano
        barmode='group',  # Barras agrupadas por ano
        title='Distribuição de Alunos por Pedra-Conceito ao Longo dos Anos',
        text='TotalAlunos_Texto',  # Usar a coluna formatada como texto nas barras
        category_orders={'PEDRA': ['Quartzo', 'Ágata', 'Ametista', 'Topázio']}  # Ordem desejada
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Pedra',
        yaxis_title='Total de Alunos',
        yaxis={'range': [0, 450]},
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='ANO',  # Título da legenda
        showlegend=True  # Garantir que a legenda seja exibida
    )

    # Ajustar a formatação da legenda e adicionar rótulos nas barras
    fig.update_traces(
        textposition='outside'  # Posicionar os rótulos das barras fora delas
    )

    # Exibir o gráfico
    fig.show()

    ##-------------------------------------------------------------------------------
    ## **Análise da Distribuição de Alunos por Pedra-Conceito e Ano**
    # Agrupar por PEDRA e ANO, contar o número de alunos
    df_pedra_ano = df_concat.groupby(['PEDRA', 'ANO']).size().reset_index(name='Total_Alunos')

    # Calcular o total geral de alunos por ano
    total_alunos_por_ano = df_pedra_ano.groupby('ANO')['Total_Alunos'].sum().reset_index(name='Total_Geral')

    # Mesclar o total geral com o DataFrame principal
    df_pedra_ano = pd.merge(df_pedra_ano, total_alunos_por_ano, on='ANO')

    # Calcular a porcentagem de alunos por pedra em relação ao total do ano
    df_pedra_ano['Porcentagem'] = (df_pedra_ano['Total_Alunos'] / df_pedra_ano['Total_Geral']) * 100

    df_pedra_ano

    # Criar o gráfico de rosca
    fig = px.pie(
        df_pedra_ano,
        names='PEDRA',  # Categorias: tipos de pedra
        values='Porcentagem',  # Valores: porcentagem de alunos
        color='PEDRA',  # Cores diferentes para cada pedra
        title='Distribuição de Alunos por Pedra-Conceito',
        hole=0.4,  # Define o tamanho do buraco no meio (gráfico de rosca)
        facet_col='ANO',  # Separar por ano (opcional: small multiples)
        labels={'Porcentagem': 'Porcentagem de Alunos'},  # Rótulos personalizados
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        legend_title_text='Pedra',  # Título da legenda
        showlegend=True,  # Garantir que a legenda seja exibida

    )

    # Exibir o gráfico
    fig.show()
    ##-------------------------------------------------------------------------------
    ## **Análise da Distribuição de Alunos por Fase, Pedra e Ano**
    # Supondo que df_concat já esteja preparado e limpo
    # Remove linhas onde FASE ou PEDRA são NaN
    df_fase_pedra = df_concat[['FASE', 'PEDRA', 'ANO']].dropna(subset=['FASE', 'PEDRA'])

    # Agrupa por FASE, ANO e PEDRA, contando o número de alunos
    df_agrupado = (
        df_fase_pedra.groupby(['FASE', 'ANO', 'PEDRA'])
        .size()
        .reset_index(name='Quantidade')
    )

    # Calcula o total de alunos por FASE e ANO para exibir nos rótulos
    df_total = (
        df_agrupado.groupby(['FASE', 'ANO'])['Quantidade']
        .sum()
        .reset_index(name='Total_Alunos')
    )
    df_agrupado = pd.merge(df_agrupado, df_total, on=['FASE', 'ANO'], how='left')

    # Cria uma nova coluna com o texto formatado para os rótulos (quantidade + total)
    df_agrupado['Texto_Rotulo'] = (
        df_agrupado['Quantidade'].astype(str) 
    )

    df_agrupado

    # Cria um gráfico de barras empilhadas
    fig = px.bar(
        df_agrupado,
        x='FASE',  # Eixo X: fases
        y='Quantidade',  # Eixo Y: quantidade de alunos
        color='PEDRA',  # Cores diferentes para cada PEDRA
        barmode='stack',  # Barras empilhadas por PEDRA
        facet_col='ANO',  # Separa os gráficos por ano
        facet_col_wrap=1,  # Exibe os gráficos um abaixo do outro
        title='Distribuição de Alunos por Fase, Pedra e Ano',
        labels={'Quantidade': 'Número de Alunos', 'FASE': 'Fase', 'PEDRA': 'Pedra'},
        text='Texto_Rotulo',  # Exibe o rótulo com a quantidade e o total
        category_orders={'PEDRA': ['Quartzo', 'Ágata', 'Ametista', 'Topázio']}  # Ordem desejada
    )

    # Ajustar o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Total de Alunos',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},  # Ordenar fases em ordem crescente
        yaxis={'range': [0, 220]},
        bargap=0.2,  # Espaçamento entre as barras de diferentes fases
        bargroupgap=0.1,  # Espaçamento entre as barras do mesmo grupo (ano)
        legend_title_text='ANO',  # Título da legenda
        showlegend=True,  # Garantir que a legenda seja exibida
        height=1200  # Aumentar a altura do gráfico
    )

    # Ajusta a posição dos rótulos para ficarem acima das barras
    fig.update_traces(textposition='outside')

    # Exibe o gráfico
    fig.show()

    ##-------------------------------------------------------------------------------
    ## **Distribuição de alunos que alcançaram ou não o Ponto de Virada por Pedra e Ano**
    # Remove linhas onde PONTO_VIRADA é NaN
    df_ponto_virada = df_concat[['PONTO_VIRADA', 'PEDRA', 'ANO']].dropna(subset=['PONTO_VIRADA'])

    # Converte PONTO_VIRADA para string, caso ainda não esteja
    df_ponto_virada['PONTO_VIRADA'] = df_ponto_virada['PONTO_VIRADA'].astype(str)

    # Agrupa por PEDRA, ANO e PONTO_VIRADA, contando o número de alunos
    df_agrupado = (
        df_ponto_virada.groupby(['PEDRA', 'ANO', 'PONTO_VIRADA'])
        .size()
        .reset_index(name='Quantidade')
    )

    # Calcula o total de alunos por PEDRA e ANO para exibir nos rótulos
    df_total = (
        df_agrupado.groupby(['PEDRA', 'ANO'])['Quantidade']
        .sum()
        .reset_index(name='Total_Alunos')
    )
    df_agrupado = pd.merge(df_agrupado, df_total, on=['PEDRA', 'ANO'], how='left')

    # Cria uma nova coluna com o texto formatado para os rótulos (quantidade + total)
    df_agrupado['Texto_Rotulo'] = (
        df_agrupado['Quantidade'].astype(str)
    )

    df_agrupado

    # Cria um gráfico de barras agrupadas
    fig = px.bar(
        df_agrupado,
        x='PEDRA',  # Eixo X: tipos de pedra
        y='Quantidade',  # Eixo Y: quantidade de alunos
        color='PONTO_VIRADA',  # Cores diferentes para "Sim" e "Não"
        barmode='group',  # Barras agrupadas por PONTO_VIRADA
        facet_col='ANO',  # Separa os gráficos por ano
        title='Distribuição de alunos que alcançaram ou não o Ponto de Virada por Pedra e Ano',
        labels={'Quantidade': 'Número de Alunos', 'PONTO_VIRADA': 'Alcançou Ponto de Virada'},
        text='Texto_Rotulo',  # Exibe o rótulo com a quantidade e o total
        category_orders={'PEDRA': ['Quartzo', 'Ágata', 'Ametista', 'Topázio']},  # Ordem desejada
    )

    # Ajusta o layout para melhorar a visualização
    fig.update_layout(
        xaxis_title='Pedra',
        yaxis_title='Número de Alunos',
        legend_title_text='Alcançou Ponto de Virada',
        showlegend=True,
    )

    # Ajusta a posição dos rótulos para ficarem acima das barras
    fig.update_traces(textposition='outside')

    # Exibe o gráfico
    fig.show()

