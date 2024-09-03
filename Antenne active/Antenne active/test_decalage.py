# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:47:04 2024

@author: IvanB
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:16:39 2024

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
import time


## Paramètres 

#ADC
Vpp = 2                                                      # Tension du DAC dans la boucle
enob = 4                                                     # Facteur de quantification
upsampling = 64                                              # Facteur de suréchantillonnage
fs = 60e9                                                    # Fréquence d'échantillonnage (60GHz)
f_tx = 20e9                                                  # Fréquence porteuse
time_s =  0.00000001                                          # Durée du signal
n_samples = round(fs*time_s)                                 # Nombre d'échantillons
cutoff_freq = 200                                            # Fréquence de coupure du filtre pass-bas
samples_per_symbol = 10                                      # Nombre d'échantillons par symbole (30 - 60 - 90)
c = 299792458                                                # Vitesse lumière dans le vide
lambda0 = c / (f_tx)                                         # Longueur d'onde    
t = np.linspace(0, time_s, n_samples*samples_per_symbol)     # Vecteur temps


#Antenne
N = 20              # Nombre d'éléments dans chaque dimension
d = 0.5 * lambda0   # Espacement entre les éléments en longueur d'onde


#Steering
dU = 0.1
dV = 0
theta = np.arctan(np.sqrt((dU**2 + dV**2)))
phi = np.arctan2(dU, dV) 
theta_deg = np.rad2deg(theta)
phi_deg = np.rad2deg(phi)




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

@njit
def calculate_time(phases):
    k = 2 * np.pi / lambda0
    time_delay = phases / (k*c)
    return time_delay






## 1- Signal de base recu

noise = create_awgn(n_samples*samples_per_symbol, bandwidth=0.1)
signal_qpsk = qpsk(n_samples,0.2,samples_per_symbol)
signal_qpsk_conv = signal_qpsk[0]
signal_qpsk_conv_noise = signal_qpsk_conv + noise
fft_qpsk = np.fft.fft(signal_qpsk_conv_noise)

signal_in = frequency_offset(signal_qpsk_conv_noise,fs/10,fs) 
fft_qpsk_offset = np.fft.fft(signal_in)




## 2- Signal numérisé par l'ADC

signal_out = ADC_QPSK_GLOBAL(signal_in)
fft_out = np.fft.fft(signal_out)




## 3- Décalage de phase pour les différents éléments à un angle fixé


X,Y = generate_positions(N,d)
phases = calculate_phases(X, Y, theta, phi)
phases_factor = np.exp(1j * phases)
number_of_wrapping_phase = np.floor(phases/ (2*np.pi))
max_wrapping = np.max(np.abs(number_of_wrapping_phase))

time_delay = calculate_time(phases)
one_sample_time = time_s/len(t)
number_of_sample_time = np.round(time_delay/one_sample_time)
max_sample_time = int(np.max(np.abs(number_of_sample_time)))

signaux_phase = np.zeros((n_samples*samples_per_symbol, N, N), dtype=complex)

for i in range(n_samples*samples_per_symbol):
    signaux_phase[i, :, :] = signal_out[i]
    signaux_phase[i, :, :] *= phases_factor




i = 10
j = 10

signal_QPSK = signaux_phase[:,i,j]

hrrc = signal_qpsk[2]
symbols = signal_qpsk[3]
hrrc_inv = np.flipud(hrrc)
demod_convolv = signal.convolve(signal_QPSK, hrrc_inv,mode='same')
symbols_out_demod = demod_convolv[0::samples_per_symbol]

angle_diff = np.angle(symbols_out_demod * np.conj(symbols))
angle_moyen = np.mean(angle_diff)
symbols_out_demod_egal = symbols_out_demod * np.exp(-1j * angle_moyen)

plot_IQ_symbols(symbols_out_demod_egal, title="Constellation qpsk sur un element") 





one_sample_time = 1 / fs  

fft_signaux_phase = np.fft.fft(signal_QPSK)
f = np.fft.fftfreq(len(signal_QPSK), d=1/fs)

fft_signaux_phase_decale = fft_signaux_phase * np.exp(-2j * np.pi * f *10*one_sample_time)

a_decale = np.fft.ifft(fft_signaux_phase_decale)

a_normal = signal_QPSK




plt.figure(figsize=(12, 6))
plt.plot(np.real(a_normal), label="Signal original")
plt.plot(np.real(a_decale), label="Signal décalé")

plt.grid()
plt.show()


























