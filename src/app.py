import pathlib
import dash
import openpyxl
from dash import dcc, html, Input, Output, State
from dash.dependencies import ALL
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time

app = dash.Dash(__name__)
server = app.server

PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("data").resolve()

path = "Competencias_Sabana_15042024.xlsx"

competencias = pd.read_excel(DATA_PATH.joinpath(path))

model = SentenceTransformer("all-MiniLM-L6-v2")

# Establecer el número inicial de inputs de oración
initial_input_count = 3

app.layout = html.Div([
    html.H1("Búsqueda de Similitud"),
    html.Div(id="input-container", children=[
        html.Div([
            dcc.Input(id={'type': 'input-sentence', 'index': i}, type='text', placeholder=f'Ingrese la oración {i+1}')
            for i in range(initial_input_count)
        ]),
        html.Button('Agregar más', id='button-add-input', n_clicks=0),
        dcc.Input(id='input-num-top', type='number', placeholder='Número de similitudes', value=3),
        html.Button('Buscar', id='button-search', n_clicks=0),
    ]),
    html.Div(id='output-container', children=[]),
    html.Div(id='progress-container', children=[
        dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
        html.Div(id="progress-bar-container", children=[
            html.Div(id="progress-bar", style={"width": "0%"})
        ])
    ])
])

# Callback para agregar más inputs de oración cuando se hace clic en el botón "Agregar más"
@app.callback(
    Output('input-container', 'children'),
    [Input('button-add-input', 'n_clicks')],
    [State('input-container', 'children')]
)
def add_input(n_clicks, existing_children):
    if n_clicks > 0:
        new_input = dcc.Input(id={'type': 'input-sentence', 'index': len(existing_children)}, type='text', placeholder=f'Ingrese la oración {len(existing_children)+1}')
        existing_children.append(html.Div(new_input))
    return existing_children

@app.callback(
    Output('output-container', 'children'),
    [Input('button-search', 'n_clicks')],
    [State({'type': 'input-sentence', 'index': ALL}, 'value'),
     State('input-num-top', 'value')]
)
def update_output(n_clicks, sentences, num_top_similitudes):
    if n_clicks > 0:
        embeddings = []
        competencias_embeddings = []
        programas = []
        campos = []

        for _, row in competencias.iterrows():
            texto = row["Competencia"]
            programa = row["Programa"]
            campo = row["Campo 1"]
            embedding = model.encode(texto, convert_to_tensor=True)
            embeddings.append(embedding)
            competencias_embeddings.append(texto)
            programas.append(programa)
            campos.append(campo)
            
        all_results = []

        for sentence in sentences:
            if sentence:
                embedding_2 = model.encode(sentence, convert_to_tensor=True)

                similitudes = []

                for i in range(len(embeddings)):
                    similarity = util.pytorch_cos_sim(embeddings[i], embedding_2)
                    similitudes.append((i, similarity))

                similitudes_ordenadas = sorted(similitudes, key=lambda x: x[1], reverse=True)
                top_similitudes = similitudes_ordenadas[:num_top_similitudes]

                resultados = []

                for index, similarity in top_similitudes:
                    competencia = competencias_embeddings[index]
                    programa = programas[index]
                    campo = campos[index]
                    similitud = similarity[0][0]
                    resultados.append(html.Div([html.Strong(f'Competencia: {competencia}'), html.Br(),
                                                html.Span(f'Programa: {programa}'), html.Br(),
                                                html.Span(f'Campo: {campo}'), html.Br(),
                                                html.Span(f'Similitud: {round(similitud.item()*100, 2)} %')]))

                all_results.append(html.Div([
                    html.H3(f'Oración: {sentence}'),
                    *resultados
                ]))

        return all_results

    return []

if __name__ == '__main__':
    app.run_server(debug=False)
