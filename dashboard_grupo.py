import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from codigo import filter_columns, cleaning_dataset, padronizar_colunas  # Import the functions
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('PEDE_PASSOS_DATASET_FIAP.csv', delimiter=';')

# ...existing data processing functions...

# Process the dataset
df_2020 = filter_columns(df, ['2021', '2022'])
df_2020 = cleaning_dataset(df_2020)
df_2020['FASE_2020'] = df_2020['FASE_TURMA_2020'].str.extract(r'(\d+)')
df_2020['TURMA_2020'] = df_2020['FASE_TURMA_2020'].str.extract(r'(\D+)')
df_2020 = df_2020.drop(df_2020[df_2020['FASE_2020'] == '207'].index)
df_2020 = padronizar_colunas(df_2020)
df_2020["ANO"] = 2020

df_2021 = filter_columns(df, ['2020', '2022'])
df_2021 = cleaning_dataset(df_2021)
df_2021 = padronizar_colunas(df_2021)
df_2021["ANO"] = 2021

df_2022 = filter_columns(df, ['2020', '2021'])
df_2022 = cleaning_dataset(df_2022)
df_2022.rename(columns={'REC_AVA_1_2022': 'REC_EQUIPE_1_2022', 'REC_AVA_2_2022': 'REC_EQUIPE_2_2022', 'REC_AVA_3_2022': 'REC_EQUIPE_3_2022', 'REC_AVA_4_2022': 'REC_EQUIPE_4_2022'}, inplace=True)
df_2022 = padronizar_colunas(df_2022)
df_2022["ANO"] = 2022

df_concat = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)
df_concat.drop('FASE_TURMA', axis=1, inplace=True)
df_concat.replace('#NULO!', np.nan, inplace=True)
df_concat['IAA'] = df_concat['IAA'].astype(float).round(2)
df_concat['IAN'] = df_concat['IAN'].astype(float).round(2)
df_concat['IDA'] = df_concat['IDA'].astype(float).round(2)
df_concat['IEG'] = df_concat['IEG'].astype(float).round(2)
df_concat['INDE'] = df_concat['INDE'].astype(float).round(2)
df_concat['IPP'] = df_concat['IPP'].astype(float).round(2)
df_concat['IPS'] = df_concat['IPS'].astype(float).round(2)
df_concat['IPV'] = df_concat['IPV'].astype(float).round(2)
df_concat['FASE'] = df_concat['FASE'].astype(float).round(0).astype(int)

# Initialize the Dash app

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', "https://passosmagicos.org.br/wp-content/themes/Divi/style.css?ver=4.0.9"]
app = dash.Dash(external_stylesheets=external_stylesheets)
# app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Header(
        html.Div([
            html.Img(src='https://passosmagicos.org.br/wp-content/uploads/2020/10/Passos-magicos-icon-cor.png', className='logo', width=250),
            html.H1("ANÁLISE DOS INDICADORES DE 2020-2022")
        ], style={"background-color": "#ecf2fb"}), style={"background-color": "#ecf2fb"}),
    # html.Div(
    # html.Header(
    #     html.Div([
    #         html.Img(src='https://passosmagicos.org.br/wp-content/uploads/2020/10/Passos-magicos-icon-cor.png', className='logo', width=250),
    #         html.H1("ANÁLISE DOS INDICADORES DE 2020-2022"),
    #     ]),
    # style={"background-color": "#ecf2fb"}), style={"background-color": "#ecf2fb"}),
    dcc.Tabs([
        dcc.Tab(label='Total Alunos por Fase e Ano', children=[
            dcc.Graph(id='total-alunos-fase-ano')
        ]),
        dcc.Tab(label='Comparação de Indicadores', children=[
            dcc.Dropdown(
                id='indicador-dropdown',
                options=[{'label': i, 'value': i} for i in ['IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']],
                value='IAA'
            ),
            dcc.Graph(id='comparacao-indicadores')
        ]),
        dcc.Tab(label='Recomendações Equipe', children=[
            dcc.Graph(id='recomendacoes-equipe1'),
            dcc.Graph(id='recomendacoes-equipe2'),
            dcc.Graph(id='recomendacoes-equipe3'),
            dcc.Graph(id='recomendacoes-equipe4')
        ]),
        dcc.Tab(label='Distribuição de Alunos por Pedra-Conceito', children=[
            dcc.Graph(id='distribuicao-pedra-conceito')
        ]),
        dcc.Tab(label='Distribuição de Alunos por Fase, Pedra e Ano', children=[
            dcc.Graph(id='distribuicao-fase-pedra-ano')
        ]),
        dcc.Tab(label='Distribuição de Alunos por Ponto de Virada', children=[
            dcc.Graph(id='distribuicao-ponto-virada')
        ]),
        dcc.Tab(label='Boxplots por Indicador', children=[
            dcc.Dropdown(
                id='boxplot-indicador-dropdown',
                options=[{'label': i, 'value': i} for i in ['IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']],
                value='IAA'
            ),
            dcc.Graph(id='boxplot-indicador')
        ]),
        dcc.Tab(label='Correlação de Variáveis', children=[
            dcc.Graph(id='correlacao-variaveis')
        ]),
        dcc.Tab(label='Representatividade de Alunos Promovidos', children=[
            dcc.Graph(id='representatividade-promovidos-equipe1'),
            dcc.Graph(id='representatividade-promovidos-equipe2'),
            dcc.Graph(id='representatividade-promovidos-equipe3'),
            dcc.Graph(id='representatividade-promovidos-equipe4')
        ]),
        dcc.Tab(label='Distribuição de Alunos por Pedra-Conceito e Ano', children=[
            dcc.Graph(id='distribuicao-pedra-conceito-ano')
        ]),
        dcc.Tab(label='Distribuição de Alunos por Fase, Pedra e Ano', children=[
            dcc.Graph(id='distribuicao-alunos-fase-pedra-ano')
        ]),
        dcc.Tab(label='Distribuição de Alunos por Ponto de Virada e Pedra', children=[
            dcc.Graph(id='distribuicao-ponto-virada-pedra-ano')
        ])
    ])
])

# Define the callbacks to update the graphs
@app.callback(
    Output('total-alunos-fase-ano', 'figure'),
    Input('total-alunos-fase-ano', 'id')
)
def update_total_alunos_fase_ano(_):
    total_alunos = df_concat.groupby(['FASE', 'ANO']).size().reset_index(name='Total_Alunos')
    total_alunos['TotalAlunos_Texto'] = total_alunos['Total_Alunos'].round(0).astype(str)
    total_alunos['ANO'] = total_alunos['ANO'].astype(str)
    total_alunos = total_alunos.sort_values(by='FASE')
    fig = px.bar(
        total_alunos,
        x='FASE',
        y='Total_Alunos',
        color='ANO',  # Corrigir 'cor' para 'color'
        barmode='group',  # Corrigir 'grupo' para 'group'
        title='Total de Alunos por Fase e Ano',
        text='TotalAlunos_Texto'
    )
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Total de Alunos',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 220]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='ANO',
        showlegend=True
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('comparacao-indicadores', 'figure'),
    Input('indicador-dropdown', 'value')
)
def update_comparacao_indicadores(indicador):
    comparacao_fase = df_concat.groupby(['FASE', 'ANO'])[['IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']].mean().reset_index()
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)
    comparacao_fase['ANO'] = comparacao_fase['ANO'].astype(str)
    comparacao_fase = comparacao_fase.sort_values(by='FASE', key=lambda x: x.astype(int))
    media_total_ano = df_concat.groupby('ANO')[['IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV']].mean().reset_index()
    media_total_ano['FASE'] = 'Média Total'
    comparacao_fase = pd.concat([comparacao_fase, media_total_ano], ignore_index=True)
    comparacao_fase['FASE'] = comparacao_fase['FASE'].astype(str)
    comparacao_fase['ANO'] = comparacao_fase['ANO'].astype(str)
    comparacao_fase = comparacao_fase.sort_values(by=['FASE', 'ANO'])
    comparacao_fase[f'{indicador}_Texto'] = comparacao_fase[indicador].round(2).astype(str)
    fig = px.bar(
        comparacao_fase,
        x='FASE',
        y=indicador,
        color='ANO',
        barmode='group',
        title=f'Comparação do {indicador} por Fase e Ano',
        text=f'{indicador}_Texto'
    )
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title=f'Média do {indicador}',
        yaxis={'range': [0, 11]},
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        showlegend=True
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('recomendacoes-equipe1', 'figure'),
    Input('recomendacoes-equipe1', 'id')
)
def update_recomendacoes_equipe1(_):
    total_recomendacao_equipe1 = df_concat.groupby(['REC_EQUIPE_1', 'ANO']).size().reset_index(name='count')
    total_recomendacao_equipe1['Total_Alunos_Ano'] = total_recomendacao_equipe1.groupby('ANO')['count'].transform('sum')
    total_recomendacao_equipe1['Represent'] = (total_recomendacao_equipe1['count'] / total_recomendacao_equipe1['Total_Alunos_Ano']) * 100
    total_recomendacao_equipe1['Represent_Texto'] = total_recomendacao_equipe1['Represent'].astype(float).round(2).astype(str) + '%'
    total_recomendacao_equipe1 = total_recomendacao_equipe1.sort_values(by='REC_EQUIPE_1')
    fig = px.bar(
        total_recomendacao_equipe1,
        x='REC_EQUIPE_1',
        y='Represent',
        color='ANO',
        barmode='group',
        title='Representatividade de Alunos por Recomendação Equipe 1',
        text='Represent_Texto'
    )
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 70]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        showlegend=True
    )
    fig.update_traces(textposition='inside')
    # fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('recomendacoes-equipe2', 'figure'),
    Input('recomendacoes-equipe2', 'id')
)
def update_recomendacoes_equipe2(_):
    total_recomendacao_equipe2 = df_concat.groupby(['REC_EQUIPE_2', 'ANO']).size().reset_index(name='count')
    total_recomendacao_equipe2['Total_Alunos_Ano'] = total_recomendacao_equipe2.groupby('ANO')['count'].transform('sum')
    total_recomendacao_equipe2['Represent'] = (total_recomendacao_equipe2['count'] / total_recomendacao_equipe2['Total_Alunos_Ano']) * 100
    total_recomendacao_equipe2['Represent_Texto'] = total_recomendacao_equipe2['Represent'].astype(float).round(2).astype(str) + '%'
    total_recomendacao_equipe2 = total_recomendacao_equipe2.sort_values(by='REC_EQUIPE_2')
    fig = px.bar(
        total_recomendacao_equipe2,
        x='REC_EQUIPE_2',
        y='Represent',
        color='ANO',
        barmode='group',
        title='Representatividade de Alunos por Recomendação Equipe 2',
        text='Represent_Texto'
    )
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 70]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        showlegend=True
    )
    fig.update_traces(textposition='inside')
    return fig

@app.callback(
    Output('recomendacoes-equipe3', 'figure'),
    Input('recomendacoes-equipe3', 'id')
)
def update_recomendacoes_equipe3(_):
    total_recomendacao_equipe3 = df_concat.groupby(['REC_EQUIPE_3', 'ANO']).size().reset_index(name='count')
    total_recomendacao_equipe3['Total_Alunos_Ano'] = total_recomendacao_equipe3.groupby('ANO')['count'].transform('sum')
    total_recomendacao_equipe3['Represent'] = (total_recomendacao_equipe3['count'] / total_recomendacao_equipe3['Total_Alunos_Ano']) * 100
    total_recomendacao_equipe3['Represent_Texto'] = total_recomendacao_equipe3['Represent'].astype(float).round(2).astype(str) + '%'
    total_recomendacao_equipe3 = total_recomendacao_equipe3.sort_values(by='REC_EQUIPE_3')
    fig = px.bar(
        total_recomendacao_equipe3,
        x='REC_EQUIPE_3',
        y='Represent',
        color='ANO',
        barmode='group',
        title='Representatividade de Alunos por Recomendação Equipe 3',
        text='Represent_Texto'
    )
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 70]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        showlegend=True
    )
    fig.update_traces(textposition='inside')
    return fig

@app.callback(
    Output('recomendacoes-equipe4', 'figure'),
    Input('recomendacoes-equipe4', 'id')
)
def update_recomendacoes_equipe4(_):
    total_recomendacao_equipe4 = df_concat.groupby(['REC_EQUIPE_4', 'ANO']).size().reset_index(name='count')
    total_recomendacao_equipe4['Total_Alunos_Ano'] = total_recomendacao_equipe4.groupby('ANO')['count'].transform('sum')
    total_recomendacao_equipe4['Represent'] = (total_recomendacao_equipe4['count'] / total_recomendacao_equipe4['Total_Alunos_Ano']) * 100
    total_recomendacao_equipe4['Represent_Texto'] = total_recomendacao_equipe4['Represent'].astype(float).round(2).astype(str) + '%'
    total_recomendacao_equipe4 = total_recomendacao_equipe4.sort_values(by='REC_EQUIPE_4')
    fig = px.bar(
        total_recomendacao_equipe4,
        x='REC_EQUIPE_4',
        y='Represent',
        color='ANO',
        barmode='group',
        title='Representatividade de Alunos por Recomendação Equipe 4',
        text='Represent_Texto'
    )
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 90]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        showlegend=True
    )
    fig.update_traces(textposition='inside')
    return fig

@app.callback(
    Output('distribuicao-pedra-conceito', 'figure'),
    Input('distribuicao-pedra-conceito', 'id')
)
def update_distribuicao_pedra_conceito(_):
    total_alunos = df_concat.groupby(['PEDRA', 'ANO']).size().reset_index(name='Total_Alunos')
    total_alunos['TotalAlunos_Texto'] = total_alunos['Total_Alunos'].round(0).astype(str)
    total_alunos['ANO'] = total_alunos['ANO'].astype(str)
    total_alunos = total_alunos.sort_values(by='PEDRA')
    fig = px.bar(
        total_alunos,
        x='PEDRA',
        y='Total_Alunos',
        color='ANO',
        barmode='group',
        title='Distribuição de Alunos por Pedra-Conceito ao Longo dos Anos',
        text='TotalAlunos_Texto',
        category_orders={'PEDRA': ['Quartzo', 'Ágata', 'Ametista', 'Topázio']}
    )
    fig.update_layout(
        xaxis_title='Pedra',
        yaxis_title='Total de Alunos',
        yaxis={'range': [0, 450]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='ANO',
        showlegend=True
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('distribuicao-fase-pedra-ano', 'figure'),
    Input('distribuicao-fase-pedra-ano', 'id')
)
def update_distribuicao_fase_pedra_ano(_):
    df_fase_pedra = df_concat[['FASE', 'PEDRA', 'ANO']].dropna(subset=['FASE', 'PEDRA'])
    df_agrupado = df_fase_pedra.groupby(['FASE', 'ANO', 'PEDRA']).size().reset_index(name='Quantidade')
    df_total = df_agrupado.groupby(['FASE', 'ANO'])['Quantidade'].sum().reset_index(name='Total_Alunos')
    df_agrupado = pd.merge(df_agrupado, df_total, on=['FASE', 'ANO'], how='left')
    df_agrupado['Texto_Rotulo'] = df_agrupado['Quantidade'].astype(str)
    fig = px.bar(
        df_agrupado,
        x='FASE',
        y='Quantidade',
        color='PEDRA',
        barmode='stack',
        facet_col='ANO',
        # facet_col_wrap=1,
        title='Distribuição de Alunos por Fase, Pedra e Ano',
        labels={'Quantidade': 'Número de Alunos', 'FASE': 'Fase', 'PEDRA': 'Pedra'},
        text='Texto_Rotulo',
        category_orders={'PEDRA': ['Quartzo', 'Ágata', 'Ametista', 'Topázio']}
    )
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Total de Alunos',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        # yaxis={'range': [0, 90]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        margin=dict(autoexpand=True),
        showlegend=True
        
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('distribuicao-ponto-virada', 'figure'),
    Input('distribuicao-ponto-virada', 'id')
)
def update_distribuicao_ponto_virada(_):
    df_ponto_virada = df_concat[['PONTO_VIRADA', 'PEDRA', 'ANO']].dropna(subset=['PONTO_VIRADA'])
    df_ponto_virada['PONTO_VIRADA'] = df_ponto_virada['PONTO_VIRADA'].astype(str)
    df_agrupado = df_ponto_virada.groupby(['PEDRA', 'ANO', 'PONTO_VIRADA']).size().reset_index(name='Quantidade')
    df_total = df_agrupado.groupby(['PEDRA', 'ANO'])['Quantidade'].sum().reset_index(name='Total_Alunos')
    df_agrupado = pd.merge(df_agrupado, df_total, on=['PEDRA', 'ANO'], how='left')
    df_agrupado['Texto_Rotulo'] = df_agrupado['Quantidade'].astype(str)
    fig = px.bar(
        df_agrupado,
        x='PEDRA',
        y='Quantidade',
        color='PONTO_VIRADA',
        barmode='group',
        facet_col='ANO',
        title='Distribuição de alunos que alcançaram ou não o Ponto de Virada por Pedra e Ano',
        labels={'Quantidade': 'Número de Alunos', 'PONTO_VIRADA': 'Alcançou Ponto de Virada'},
        text='Texto_Rotulo',
        category_orders={'PEDRA': ['Quartzo', 'Ágata', 'Ametista', 'Topázio']}
    )
    fig.update_layout(
        # xaxis_title='Pedra',
        yaxis_title='Número de Alunos',
        legend_title_text='Alcançou Ponto de Virada',
        showlegend=True
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('boxplot-indicador', 'figure'),
    Input('boxplot-indicador-dropdown', 'value')
)
def update_boxplot_indicador(indicador_2):
    anos = df_concat['ANO'].unique()
    # fig = make_subplots(rows=1, cols=len(anos), subplot_titles=[f'Box Plot - {indicador_2} por Fase - {ano}' for ano in anos])
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f'Box Plot - {indicador_2} por Fase - {ano}' for ano in anos])
    for i, ano in enumerate(anos):
        df_filtrado = df_concat[df_concat['ANO'] == ano]
        boxplot = go.Box(
            x=df_filtrado['FASE'],
            y=df_filtrado[indicador_2],
            name=str(ano)
        )
        fig.add_trace(boxplot, row=1, col=i+1)
    fig.update_layout(
        title_text=f"Boxplots de {indicador_2} por Fase e Ano",
        showlegend=False,
        height=400,
        width=1200
    )
    return fig

@app.callback(
    Output('correlacao-variaveis', 'figure'),
    Input('correlacao-variaveis', 'id')
)
def update_correlacao_variaveis(a):
    df_promovido = df_concat[
        (df_concat['REC_EQUIPE_1'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])) |
        (df_concat['REC_EQUIPE_2'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])) |
        (df_concat['REC_EQUIPE_3'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])) |
        (df_concat['REC_EQUIPE_4'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa']))
    ]
    df_promovido.loc[:, 'Turma_'] = df_promovido['TURMA'].astype('category').cat.codes
    variaveis = ['IAA', 'IAN', 'IDA', 'IEG', 'INDE', 'IPP', 'IPS', 'IPV', 'FASE', 'Turma_']
    df_correlacao = df_promovido[variaveis]
    matriz_correlacao = df_correlacao.corr()
    fig = px.imshow(matriz_correlacao, text_auto=True, aspect="auto", title='Matriz de Correlação - Alunos Promovidos')
    return fig

@app.callback(
    Output('representatividade-promovidos-equipe1', 'figure'),
    Input('representatividade-promovidos-equipe1', 'id')
)
def update_representatividade_promovidos_equipe1(b):
    total_alunos = df_concat.groupby(['FASE', 'ANO']).size().reset_index(name='Total_Alunos')
    df_filtered_eq1 = df_concat[df_concat['REC_EQUIPE_1'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])]
    df_grouped_eq1 = df_filtered_eq1.groupby(['ANO', 'FASE']).size().reset_index(name='count')
    df_grouped_eq1 = pd.merge(df_grouped_eq1, total_alunos, on=['ANO', 'FASE'], how='left')
    df_grouped_eq1['Represent'] = (df_grouped_eq1['count'] / df_grouped_eq1['Total_Alunos']) * 100
    df_grouped_eq1['ANO'] = df_grouped_eq1['ANO'].astype(str)
    df_grouped_eq1['Represent_Texto'] = df_grouped_eq1['Represent'].astype(float).round(2).astype(str) + '%'
    df_grouped_eq1 = df_grouped_eq1.sort_values(by='FASE')
    fig = px.bar(
        df_grouped_eq1,
        x='FASE',
        y='Represent',
        color='ANO',
        barmode='group',
        title='Representatividade de Alunos Promovidos pela Equipe 1',
        text='Represent_Texto'
    )
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 110]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        showlegend=True
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('representatividade-promovidos-equipe2', 'figure'),
    Input('representatividade-promovidos-equipe2', 'id')
)
def update_representatividade_promovidos_equipe2(c):
    total_alunos = df_concat.groupby(['FASE', 'ANO']).size().reset_index(name='Total_Alunos')
    df_filtered_eq2 = df_concat[df_concat['REC_EQUIPE_2'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])]
    df_grouped_eq2 = df_filtered_eq2.groupby(['ANO', 'FASE']).size().reset_index(name='count')
    df_grouped_eq2 = pd.merge(df_grouped_eq2, total_alunos, on=['ANO', 'FASE'], how='left')
    df_grouped_eq2['Represent'] = (df_grouped_eq2['count'] / df_grouped_eq2['Total_Alunos']) * 100
    df_grouped_eq2['ANO'] = df_grouped_eq2['ANO'].astype(str)
    df_grouped_eq2['Represent_Texto'] = df_grouped_eq2['Represent'].astype(float).round(2).astype(str) + '%'
    df_grouped_eq2 = df_grouped_eq2.sort_values(by='FASE')
    fig = px.bar(
        df_grouped_eq2,
        x='FASE',
        y='Represent',
        color='ANO',
        barmode='group',
        title='Representatividade de Alunos Promovidos pela Equipe 2',
        text='Represent_Texto'
    )
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 110]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        showlegend=True
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('representatividade-promovidos-equipe3', 'figure'),
    Input('representatividade-promovidos-equipe3', 'id')
)
def update_representatividade_promovidos_equipe3(d):
    total_alunos = df_concat.groupby(['FASE', 'ANO']).size().reset_index(name='Total_Alunos')
    df_filtered_eq3 = df_concat[df_concat['REC_EQUIPE_3'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])]
    df_grouped_eq3 = df_filtered_eq3.groupby(['ANO', 'FASE']).size().reset_index(name='count')
    df_grouped_eq3 = pd.merge(df_grouped_eq3, total_alunos, on=['ANO', 'FASE'], how='left')
    df_grouped_eq3['Represent'] = (df_grouped_eq3['count'] / df_grouped_eq3['Total_Alunos']) * 100
    df_grouped_eq3['ANO'] = df_grouped_eq3['ANO'].astype(str)
    df_grouped_eq3['Represent_Texto'] = df_grouped_eq3['Represent'].astype(float).round(2).astype(str) + '%'
    df_grouped_eq3 = df_grouped_eq3.sort_values(by='FASE')
    fig = px.bar(
        df_grouped_eq3,
        x='FASE',
        y='Represent',
        color='ANO',
        barmode='group',
        title='Representatividade de Alunos Promovidos pela Equipe 3',
        text='Represent_Texto'
    )
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 110]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        showlegend=True
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('representatividade-promovidos-equipe4', 'figure'),
    Input('representatividade-promovidos-equipe4', 'id')
)
def update_representatividade_promovidos_equipe4(dd2):
    total_alunos = df_concat.groupby(['FASE', 'ANO']).size().reset_index(name='Total_Alunos')
    df_filtered_eq4 = df_concat[df_concat['REC_EQUIPE_4'].isin(['Promovido de Fase', 'Promovido de Fase + Bolsa'])]
    df_grouped_eq4 = df_filtered_eq4.groupby(['ANO', 'FASE']).size().reset_index(name='count')
    df_grouped_eq4 = pd.merge(df_grouped_eq4, total_alunos, on=['ANO', 'FASE'], how='left')
    df_grouped_eq4['Represent'] = (df_grouped_eq4['count'] / df_grouped_eq4['Total_Alunos']) * 100
    df_grouped_eq4['ANO'] = df_grouped_eq4['ANO'].astype(str)
    df_grouped_eq4['Represent_Texto'] = df_grouped_eq4['Represent'].astype(float).round(2).astype(str) + '%'
    df_grouped_eq4 = df_grouped_eq4.sort_values(by='FASE')
    fig = px.bar(
        df_grouped_eq4,
        x='FASE',
        y='Represent',
        color='ANO',
        barmode='group',
        title='Representatividade de Alunos Promovidos pela Equipe 4',
        text='Represent_Texto'
    )
    fig.update_layout(
        xaxis_title='Recomendacao',
        yaxis_title='Representatividade',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 110]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='Ano',
        showlegend=True
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('distribuicao-pedra-conceito-ano', 'figure'),
    Input('distribuicao-pedra-conceito-ano', 'id')
)
def update_distribuicao_pedra_conceito_ano(_):
    df_pedra_ano = df_concat.groupby(['PEDRA', 'ANO']).size().reset_index(name='Total_Alunos')
    total_alunos_por_ano = df_pedra_ano.groupby('ANO')['Total_Alunos'].sum().reset_index(name='Total_Geral')
    df_pedra_ano = pd.merge(df_pedra_ano, total_alunos_por_ano, on='ANO')
    df_pedra_ano['Porcentagem'] = (df_pedra_ano['Total_Alunos'] / df_pedra_ano['Total_Geral']) * 100
    fig = px.pie(
        df_pedra_ano,
        names='PEDRA',
        values='Porcentagem',
        color='PEDRA',
        title='Distribuição de Alunos por Pedra-Conceito',
        hole=0.4,
        facet_col='ANO',
        labels={'Porcentagem': 'Porcentagem de Alunos'},
    )
    fig.update_layout(
        legend_title_text='Pedra',
        showlegend=True,
    )
    return fig

@app.callback(
    Output('distribuicao-alunos-fase-pedra-ano', 'figure'),
    Input('distribuicao-alunos-fase-pedra-ano', 'id')
)
def update_distribuicao_alunos_fase_pedra_ano(_):
    df_fase_pedra = df_concat[['FASE', 'PEDRA', 'ANO']].dropna(subset=['FASE', 'PEDRA'])
    df_agrupado = df_fase_pedra.groupby(['FASE', 'ANO', 'PEDRA']).size().reset_index(name='Quantidade')
    df_total = df_agrupado.groupby(['FASE', 'ANO'])['Quantidade'].sum().reset_index(name='Total_Alunos')
    df_agrupado = pd.merge(df_agrupado, df_total, on=['FASE', 'ANO'], how='left')
    df_agrupado['Texto_Rotulo'] = df_agrupado['Quantidade'].astype(str)
    fig = px.bar(
        df_agrupado,
        x='FASE',
        y='Quantidade',
        color='PEDRA',
        barmode='stack',
        facet_col='ANO',
        facet_col_wrap=1,
        title='Distribuição de Alunos por Fase, Pedra e Ano',
        labels={'Quantidade': 'Número de Alunos', 'FASE': 'Fase', 'PEDRA': 'Pedra'},
        text='Texto_Rotulo',
        category_orders={'PEDRA': ['Quartzo', 'Ágata', 'Ametista', 'Topázio']}
    )
    fig.update_layout(
        xaxis_title='Fase',
        yaxis_title='Total de Alunos',
        xaxis={'type': 'category', 'categoryorder': 'category ascending'},
        yaxis={'range': [0, 220]},
        bargap=0.2,
        bargroupgap=0.1,
        legend_title_text='ANO',
        showlegend=True,
        height=1200
    )
    fig.update_traces(textposition='outside')
    return fig

@app.callback(
    Output('distribuicao-ponto-virada-pedra-ano', 'figure'),
    Input('distribuicao-ponto-virada-pedra-ano', 'id')
)
def update_distribuicao_ponto_virada_pedra_ano(_):
    df_ponto_virada = df_concat[['PONTO_VIRADA', 'PEDRA', 'ANO']].dropna(subset=['PONTO_VIRADA'])
    df_ponto_virada['PONTO_VIRADA'] = df_ponto_virada['PONTO_VIRADA'].astype(str)
    df_agrupado = df_ponto_virada.groupby(['PEDRA', 'ANO', 'PONTO_VIRADA']).size().reset_index(name='Quantidade')
    df_total = df_agrupado.groupby(['PEDRA', 'ANO'])['Quantidade'].sum().reset_index(name='Total_Alunos')
    df_agrupado = pd.merge(df_agrupado, df_total, on=['PEDRA', 'ANO'], how='left')
    df_agrupado['Texto_Rotulo'] = df_agrupado['Quantidade'].astype(str)
    fig = px.bar(
        df_agrupado,
        x='PEDRA',
        y='Quantidade',
        color='PONTO_VIRADA',
        barmode='group',
        facet_col='ANO',
        title='Distribuição de alunos que alcançaram ou não o Ponto de Virada por Pedra e Ano',
        labels={'Quantidade': 'Número de Alunos', 'PONTO_VIRADA': 'Alcançou Ponto de Virada'},
        text='Texto_Rotulo',
        category_orders={'PEDRA': ['Quartzo', 'Ágata', 'Ametista', 'Topázio']}
    )
    fig.update_layout(
        xaxis_title='Pedra',
        yaxis_title='Número de Alunos',
        legend_title_text='Alcançou Ponto de Virada',
        showlegend=True
    )
    fig.update_traces(textposition='outside')
    return fig

app.run()
