import numpy as np
import scipy.signal as signal  
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, Dash
from numba import njit
import webbrowser
from usual import *  
import math



## Paramètres 

#ADC
Vpp = 2                                                 # Tension du DAC dans la boucle
enob = 4                                                # Facteur de quantification
upsampling = 64                                         # Facteur de suréchantillonnage
fs = 60e9                                               # Fréquence d'échantillonnage (60GHz)
time = 0.0000002                                         # Durée du signal
n_samples = round(fs*time)                              # Nombre d'échantillons
cutoff_freq = 200                                       # Fréquence de coupure du filtre pass-bas
samples_per_symbol = 50                                 # Nombre d'échantillons par symbole
c = 299792458                                           # Vitesse lumière dans le vide
lambda0 = c / (fs/10)                                   # Longueur d'onde    
t = np.linspace(0, time, n_samples*samples_per_symbol)  # Vecteur temps

#Antenne
N = 16              # Nombre d'éléments dans chaque dimension
d = 0.5 * lambda0   # Espacement entre les éléments en longueur d'onde

theta_deg = 34 
phi_deg = 45

theta = theta_deg*np.pi/180      
phi = phi_deg*np.pi/180



## Fonctions pour l'ADC

def lowpass_filter(signal_in):
    nyquist = fs / 2
    normal_cutoff = cutoff_freq / nyquist
    nb_coeff = 101  
    coeff_FIR = signal.firwin(nb_coeff, normal_cutoff, window='hamming')
    filtered_signal = signal.convolve(signal_in, coeff_FIR, mode='same')
    return filtered_signal

@njit
def ZOH(signal_in, upsampling_factor):
    n = len(signal_in)
    signal_out = np.zeros(n * upsampling_factor)
    for i in range(n):
        for j in range(upsampling_factor):
            signal_out[i * upsampling_factor + j] = signal_in[i]
    return signal_out

@njit
def ADC_Delta_Sigma(signal_in, Vpp, enob, upsampling):
    signal_in_up = ZOH(np.real(signal_in), upsampling)
    signal_feedback = np.zeros(len(signal_in_up))
    signal_integrator = np.zeros(len(signal_in_up))
    signal_out = np.zeros(len(signal_in_up))
    
    # Loop
    for i in np.arange(0, len(signal_in_up)):
        # Delta
        if i == 0:
            signal_feedback[i] = signal_in_up[i]
        else:
            signal_feedback[i] = signal_in_up[i] - signal_out[i-1]

        # Intégrateur
        if i == 0:
            signal_integrator[i] = signal_feedback[i]
        else:
            signal_integrator[i] = signal_feedback[i] + signal_integrator[i-1]

        signal_integrator[i] = Vpp / (2**enob) * np.round(signal_integrator[i] / (Vpp / (2**enob)))
    
        # Comparateur
        if signal_integrator[i] >= 0:
            signal_out[i] = Vpp / 2
        else:
            signal_out[i] = -Vpp / 2
            
    return signal_out

def ADC_QPSK_GLOBAL(signal_in):
    signal_out = ADC_Delta_Sigma(signal_in, Vpp, enob, upsampling)   
    signal_out_hilbert = signal.hilbert(signal_out)
    signal_out_filtered = lowpass_filter(signal_out_hilbert)
    signal_out_decimate = signal.decimate(signal_out_filtered, upsampling, ftype='fir')
    signal_out_decimate_centered = frequency_offset(signal_out_decimate, -fs / 10, fs)
    return signal_out_decimate_centered



## Fonctions pour l'antenne

def generate_positions(N, d):
    x = np.arange(N) * d
    y = np.arange(N) * d
    X, Y = np.meshgrid(x, y)
    return X, Y

@njit
def calculate_phases(X, Y, theta, phi):
    k = 2 * np.pi / lambda0
    phases = k * (X * np.sin(theta) * np.cos(phi) + Y * np.sin(theta) * np.sin(phi))
    return phases

def calculate_time(phases):
    k = 2 * np.pi / lambda0
    time_delay = 5*phases /(np.pi*fs)
    return time_delay

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



## Création de l'antenne et calcul des retards

X,Y = generate_positions(N,d)

phases = calculate_phases(X, Y, theta, phi)
phases_factor = np.exp(1j * phases)
number_of_wrapping_phase = np.floor(phases/ (2*np.pi))
max_wrapping = np.max(np.abs(number_of_wrapping_phase))

time_delay = calculate_time(phases)
one_sample_time = time/len(t)
number_of_sample_time = np.round(time_delay/one_sample_time)
max_sample_time = int(np.max(np.abs(number_of_sample_time)))



## Choix phase/time

if max_wrapping>0:
    
    print("Max wrapping de phase : ", max_wrapping, ". Il faut faire du timed array")
    
    noise = create_awgn(n_samples*samples_per_symbol, bandwidth=0.1)
    signal_qpsk = qpsk(n_samples,0.2,samples_per_symbol)
    signal_qpsk_conv = signal_qpsk[0]
    signal_qpsk_conv_noise = signal_qpsk_conv + noise
    signal_in = frequency_offset(signal_qpsk_conv_noise,fs/10,fs) 
    signal_out = ADC_QPSK_GLOBAL(signal_in)
    
    '''
    hrrc = signal_qpsk[2]
    symbols = signal_qpsk[3]
    hrrc_inv = np.flipud(hrrc)
    demod_convolv = signal.convolve(signal_out, hrrc_inv,mode='same')
    symbols_out_demod = demod_convolv[0::samples_per_symbol]
    angle_diff = np.angle(symbols_out_demod * np.conj(symbols))
    angle_moyen = np.mean(angle_diff)
    symbols_out_demod_egal = symbols_out_demod * np.exp(-1j * angle_moyen)
    plot_IQ_symbols(symbols_out_demod_egal, title="Constellation démodulée + égalisation") 
    '''
    
    signaux = np.zeros((n_samples*samples_per_symbol + 2*max_sample_time, N, N), dtype=complex)
    
    for i in range(n_samples*samples_per_symbol):
        signaux[i, :, :] = signal_out[i]
    
    for x in range(N):
        for y in range(N):
            signaux[:, x, y] = np.roll(signaux[:, x, y],max_sample_time + int(number_of_sample_time[x,y]))
            
    hrrc = signal_qpsk[2]
    symbols = signal_qpsk[3]
    hrrc_inv = np.flipud(hrrc)
    demod_convolv = signal.convolve(signaux[:, 5, 5], hrrc_inv,mode='same')
    symbols_out_demod = demod_convolv[0::samples_per_symbol]
    
    zeros_to_add = len(symbols_out_demod)-len(symbols)
    symbols = np.concatenate((symbols, np.zeros(zeros_to_add)))
    
    angle_diff = np.angle(symbols_out_demod * np.conj(symbols))
    angle_moyen = np.mean(angle_diff)
    symbols_out_demod_egal = symbols_out_demod * np.exp(-1j * angle_moyen)
    
    plot_IQ_symbols(symbols_out_demod_egal, title="Constellation de la somme des signaux en temps") 

    
    
    
else :
    
    print("Pas de wrapping de phase. Il faut faire du phased array")

    noise = create_awgn(n_samples*samples_per_symbol, bandwidth=0.1)
    signal_qpsk = qpsk(n_samples,0.2,samples_per_symbol)
    signal_qpsk_conv = signal_qpsk[0]
    signal_qpsk_conv_noise = signal_qpsk_conv + noise
    signal_in = frequency_offset(signal_qpsk_conv_noise,fs/10,fs) 
    signal_out = ADC_QPSK_GLOBAL(signal_in)
    

    signaux = np.zeros((n_samples*samples_per_symbol, N, N), dtype=complex)
    
    for i in range(n_samples):
        signaux[i, :, :] = signal_out[i]
        signaux[i, :, :] *= phases_factor



'''
# Application Dash

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Diagramme de rayonnement de l\'antenne', style={'textAlign': 'center'}),
    
    html.Div(id='max-wrapping-info', style={'textAlign': 'center', 'marginTop': '10px'}), 
    
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
     Output('phi-info', 'children'),
     Output('max-wrapping-info', 'children')],  
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
    
    # Calcul de max_wrapping
    X,Y = generate_positions(20,d)
    one_sample_time = time/len(t)

    phases = calculate_phases(X, Y, theta, phi)
    number_of_wrapping_phase = np.floor(phases/ (2*np.pi))

    times = calculate_time(phases)
    number_of_sample_time = np.round(times/one_sample_time)

    max_wrapping = np.max(np.abs(number_of_wrapping_phase))
    
    # Mise à jour des variables
    n_info = f'Nombre d\'éléments: {n}x{n}'
    theta_info = f'Angle d\'élévation: {theta_scan}°'
    phi_info = f'Angle d\'azimut: {phi_scan}°'
    
    if max_wrapping > 0:
        max_wrapping_info = f'Max wrapping de phase : {max_wrapping}. Il faut faire du timed array'
    else:
        max_wrapping_info = 'Pas de wrapping de phase. Il faut faire du phased array'

    return fig, n_info, theta_info, phi_info, max_wrapping_info

if __name__ == '__main__':
    app.run_server(port=8050, debug=True)
    webbrowser.open("http://127.0.0.1:8050/")

'''


