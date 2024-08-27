# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:37:39 2024

@author: IvanB
"""

import numpy as np
import scipy.signal as signal 
from scipy.io import loadmat 
from numba import njit
import webbrowser
from usual import *  
import math
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, Dash
import webbrowser
import io
import base64

## Paramètres 

#ADC
Vpp = 2                                                 # Tension du DAC dans la boucle
enob = 4                                                # Facteur de quantification
upsampling = 64                                         # Facteur de suréchantillonnage
fs = 60e9 
f_tx = 20e9                                             # Fréquence d'échantillonnage (60GHz)
time = 0.0000002                                        # Durée du signal
n_samples = round(fs*time)                              # Nombre d'échantillons
cutoff_freq = 200                                       # Fréquence de coupure du filtre pass-bas
samples_per_symbol = 10                                 # Nombre d'échantillons par symbole
c = 299792458                                           # Vitesse lumière dans le vide
lambda0 = c / (f_tx)                                    # Longueur d'onde    
t = np.linspace(0, time, n_samples*samples_per_symbol)  # Vecteur temps


#Antenne
N = 20              # Nombre d'éléments dans chaque dimension
d = 0.5 * lambda0   # Espacement entre les éléments en longueur d'onde

theta_deg = 50
phi_deg = 50 

theta = theta_deg*np.pi/180      
phi = phi_deg*np.pi/180

d_ts = 1200000
dU = np.tan(theta)*d_ts*np.tan(phi)/np.cos(phi)
dV = np.tan(theta)*d_ts/np.cos(phi)
dU = 0
dV = 0




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





## Signal de base recu

noise = create_awgn(n_samples*samples_per_symbol, bandwidth=0.1)
signal_qpsk = qpsk(n_samples,0.2,samples_per_symbol)
signal_qpsk_conv = signal_qpsk[0]
signal_qpsk_conv_noise = signal_qpsk_conv + noise
fft_qpsk = np.fft.fft(signal_qpsk_conv_noise)

signal_in = frequency_offset(signal_qpsk_conv_noise,fs/10,fs) 
fft_qpsk_offset = np.fft.fft(signal_in)



## Signal numérisé par l'ADC

signal_out = ADC_QPSK_GLOBAL(signal_in)
fft_out = np.fft.fft(signal_out)




## Décalage de phase pour les différents éléments à un angle fixé

X,Y = generate_positions(N,d)

phases = calculate_phases(X, Y, theta, phi)
phases_factor = np.exp(1j * phases)
number_of_wrapping_phase = np.floor(phases/ (2*np.pi))
max_wrapping = np.max(np.abs(number_of_wrapping_phase))

time_delay = calculate_time(phases)
one_sample_time = time/len(t)
number_of_sample_time = np.round(time_delay/one_sample_time)
max_sample_time = int(np.max(np.abs(number_of_sample_time)))

signaux = np.zeros((n_samples*samples_per_symbol, N, N), dtype=complex)
for i in range(n_samples*samples_per_symbol):
    signaux[i, :, :] = signal_out[i]
    signaux[i, :, :] *= phases_factor
fft_phased = np.fft.fft(signaux[:,1,1])






## GRD


nbMappingSample = 201  
thetaMax = 90  
varUV = np.sin(np.radians(thetaMax))  
u = np.linspace(-varUV, varUV, nbMappingSample)
v = u
gridU, gridV = np.meshgrid(u, v)
U = gridU
V = gridV

UU = U.reshape(1, nbMappingSample * nbMappingSample)
VV = V.reshape(1, nbMappingSample * nbMappingSample)

index_reduced = np.where(np.sqrt(UU**2 + VV**2) < 0.81)[1]
indexes = np.arange(nbMappingSample * nbMappingSample)
removed_indexes = np.setdiff1d(indexes, index_reduced)

x = np.arange(0, 361)
circle = 0.8121 * np.column_stack((np.sin(np.radians(x)), np.cos(np.radians(x))))
circle_x = circle[:, 0]
circle_y = circle[:, 1]



data = loadmat('GRD_INITIAL.mat')
GRD_INIT = data['GRD_INIT']

GRDS = np.zeros((N, N, 201, 201), dtype="complex")


for i in range(N):  
    for j in range (N):
        dUV = np.exp(2 * np.pi / lambda0 * 1j * ( X[i,j] * U + Y[i,j] * V))
        GRDS[i,j,:,:] = GRD_INIT * dUV


w = np.ones((N,N))
w = w * np.exp(1j * 2 * np.pi * (-X*dU*35  + Y*dV*35 ))
GRD_resultant = np.zeros((201, 201),dtype="complex")



# Somme des contributions de GRDS
for i in range(N):
    for j in range(N):
        GRD_resultant = GRD_resultant + GRDS[i,j,:,:] * w[i,j]

GRD_resultant_abs =np.abs(GRD_resultant)
GRD_resultant_db = 20 * np.log10(GRD_resultant_abs)



## Somme des signaux

max_index = np.argmax(GRD_resultant_abs)
k, l = np.unravel_index(max_index, GRD_resultant_abs.shape)

for i in range(N):
    for j in range(N):
        signaux[:,i,j] = signaux[:,i,j]*np.abs(w[i,j])*GRDS[i,j,k,l]


somme_signaux = np.zeros(n_samples*samples_per_symbol)

for i in range(N):
    for j in range(N):
        somme_signaux = somme_signaux +  signaux[:,i,j]




## Plot fft

plt.figure(figsize=(12, 8))
freq = np.fft.fftfreq(n_samples*samples_per_symbol, 1/fs)


plt.subplot(4, 1, 1)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq, (1/fs)*np.abs(fft_qpsk))
plt.title('FFT in')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq, (1/fs)*np.abs(fft_qpsk_offset))
plt.title('FFT in offset')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 3)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq, (1/fs)*np.abs(fft_out))
plt.title('FFT after ADC')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 4)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq, (1/fs)*np.abs(fft_phased))
plt.title('FFT phased')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')



plt.tight_layout()
plt.show()



def create_plot(dU, dV, samples_per_symbol):
    w = np.ones((N, N))
    w = w * np.exp(1j * 2 * np.pi * (-X * dU * 35 + Y * dV * 35))
    GRD_resultant = np.zeros((201, 201), dtype="complex")
    for i in range(N):
        for j in range(N):
            GRD_resultant += GRDS[i, j, :, :] * w[i, j]

    GRD_resultant_abs = np.abs(GRD_resultant)
    GRD_resultant_db = 20 * np.log10(GRD_resultant_abs)

    plt.figure()
    x = np.linspace(-1, 1, 201)
    y = np.linspace(-1, 1, 201)
    plt.pcolor(x, y, GRD_resultant_db, shading='auto')
    plt.clim(7, 50)  
    plt.colorbar()   
    plt.plot(circle[:, 1], circle[:, 0], 'r', linewidth=3)
    plt.title('GRD Resultant')
    plt.xlabel('U')
    plt.ylabel('V')

    # Sauvegarde la figure dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf





# Application Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1('Modélisation du Beamsquint', style={'textAlign': 'center'}),

    html.Div(className='container', children=[
        html.Div([
            html.Img(id='grd-graph', style={'width': '100%', 'height': '800px'}),
        ], className='graph-container'),

        html.Div([
            html.Label('dU'),
            dcc.Slider(
                id='du-slider',
                min=-1, max=1, step=0.1, value=0,
                className='dcc-slider'
            ),
            html.Label('dV'),
            dcc.Slider(
                id='dv-slider',
                min=-1, max=1, step=0.1, value=0,
                className='dcc-slider'
            ),
            html.Label('Samples per Symbol'),
            dcc.Slider(
                id='samples-slider',
                min=1, max=50, step=1, value=10,
                className='dcc-slider'
            ),
        ], className='control-panel')
    ])
])

@app.callback(
    Output('grd-graph', 'src'),
    [Input('du-slider', 'value'),
     Input('dv-slider', 'value'),
     Input('samples-slider', 'value')]
)
def update_graph(dU, dV, samples_per_symbol):
    buf = create_plot(dU, dV, samples_per_symbol)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    return "data:image/png;base64,{}".format(encoded_image)

if __name__ == '__main__':
    app.run_server(port=8050, debug=True)
    webbrowser.open("http://127.0.0.1:8050/")
