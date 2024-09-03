# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:02:06 2024

@author: IvanB
"""

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, Dash
from numba import njit
import webbrowser
from usual import *



## Paramètres 
fs = 2*1000             # Fréquence d'échantillonnage
c = 299792458           # Vitesse lumière dans le vide
lambda0 = c / (fs/10)   # Longueur d'onde    
d = 0.5*lambda0         # Espacement entre les éléments en longueur d'onde


# Générer les positions de chaque antenne
def generate_positions(N, d):
    x = np.arange(N) * d
    y = np.arange(N) * d
    X, Y = np.meshgrid(x, y)
    return X, Y


# Calcul des phases et pattern du diagramme de rayonement
@njit
def calculate_phases(X, Y, theta, phi):
    k = 2 * np.pi / lambda0
    phases = k * (X * np.sin(theta) * np.cos(phi) + Y * np.sin(theta) * np.sin(phi))
    return phases

@njit
def radiation_pattern(phases, X, Y, theta, phi):
    num_theta = 180
    num_phi = 360
    pattern = np.zeros((num_theta, num_phi), dtype=np.float64)

    for i in range(num_theta):
        for j in range(num_phi):
            kx = 2 * np.pi * np.sin(theta[i]) * np.cos(phi[j]) / lambda0
            ky = 2 * np.pi * np.sin(theta[i]) * np.sin(phi[j]) / lambda0
            exp_term = np.exp(1j * (kx * X + ky * Y - phases))
            pattern[i, j] = np.abs(np.sum(exp_term))
    
    # Normalisation
    pattern /= np.max(pattern)
    
    return pattern





# Application Dash

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Diagramme de rayonnement de l\'antenne', style={'textAlign': 'center'}),
    
    html.Div(className='container', children=[
        html.Div([
            dcc.Graph(id='graph', config={'displayModeBar': True}, style={'width': '100%', 'height': '550px'}),
        ], className='graph-container'),
        
        html.Div([
            html.Label('Nombre d\'éléments'),
            dcc.Slider(
                id='n-slider',
                min=1, max=32, value=16,
                marks={i: str(i) for i in range(1, 33)},
                className='dcc-slider'
            ),
            html.Label('Angle d\'élévation'),
            dcc.Slider(
                id='theta-slider',
                min=0, max=90, value=30,
                marks={i: str(i) for i in range(0, 91, 15)},
                className='dcc-slider'
            ),
            html.Label('Angle d\'azimut'),
            dcc.Slider(
                id='phi-slider',
                min=0, max=360, value=10,
                marks={i: str(i) for i in range(0, 361, 30)},
                className='dcc-slider'
            ),
            html.Div([
                html.H3('Variables :'),
                html.P(id='n-info', children=f'Nombre d\'éléments: 16 x 16'),
                html.P(id='theta-info', children='Angle d\'élévation: 30°'),
                html.P(id='phi-info', children='Angle d\'azimut: 10°'),
            ], className='variable-info'),
        ], className='control-panel')
    ])
])

@app.callback(
    [Output('graph', 'figure'),
     Output('n-info', 'children'),
     Output('theta-info', 'children'),
     Output('phi-info', 'children')],
    [Input('n-slider', 'value'),
     Input('theta-slider', 'value'),
     Input('phi-slider', 'value')]
)


def update_graph(n, theta_scan, phi_scan):
    
    # Calcul du nouveau diagramme
    n = int(round(n))

    X, Y = generate_positions(n, d)
    
    theta = np.radians(theta_scan)
    phi = np.radians(phi_scan)
    phases = calculate_phases(X, Y, theta, phi)
    
    theta_vals = np.linspace(0, np.pi, 180)
    phi_vals = np.linspace(0, 2 * np.pi, 360)
    pattern = radiation_pattern(phases, X, Y, theta_vals, phi_vals)
    
    pattern_dB = 20 * np.log10(pattern)
    pattern_dB[pattern_dB < -20] = np.nan
    X_pattern = pattern * np.sin(theta_vals[:, None]) * np.cos(phi_vals)
    Y_pattern = pattern * np.sin(theta_vals[:, None]) * np.sin(phi_vals)
    Z_pattern = pattern * np.cos(theta_vals[:, None])
    X_pattern[ Z_pattern <= 0 ] = 0
    Y_pattern[ Z_pattern <= 0 ] = 0
    Z_pattern[ Z_pattern <= 0 ] = 0
    
    # Création du graphique
    fig = go.Figure(data=[go.Surface(x=X_pattern, y=Y_pattern, z=Z_pattern, surfacecolor=pattern_dB, colorscale='Jet', showscale=True)])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-1, 1]),
            yaxis=dict(title='Y', range=[-1, 1]),
            zaxis=dict(title='Z', range=[-1, 1]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    # Mise à jour des variables
    n_info = f'Nombre d\'éléments: {n}x{n}'
    theta_info = f'Angle d\'élévation: {theta_scan}°'
    phi_info = f'Angle d\'azimut: {phi_scan}°'
    
    return fig, n_info, theta_info, phi_info


if __name__ == '__main__':
    app.run_server(port=8050, debug=True)
    webbrowser.open("http://127.0.0.1:8050/")