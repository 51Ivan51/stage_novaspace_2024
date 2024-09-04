# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:25:51 2024

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
from usual import *  
import math
import time




## Paramètres 

#ADC
Vpp = 2                                                      # Tension du DAC dans la boucle
enob = 4                                                     # Facteur de quantification
upsampling = 64                                              # Facteur de suréchantillonnage
fs = 60e9                                                    # Fréquence d'échantillonnage (60GHz)
f_tx = 20e9                                                  # Fréquence porteuse
time_s =  0.00000002                                         # Durée du signal
n_samples = round(fs*time_s)                                 # Nombre d'échantillons
cutoff_freq = 200                                            # Fréquence de coupure du filtre pass-bas
samples_per_symbol = 64                                      # Nombre d'échantillons par symbole (30 - 60 - 90)
c = 299792458                                                # Vitesse lumière dans le vide
lambda0 = c / (f_tx)                                         # Longueur d'onde    
t = np.linspace(0, time_s, n_samples*samples_per_symbol)     # Vecteur temps

#Antenne
N = 20              # Nombre d'éléments dans chaque dimension
d = 0.5 * lambda0   # Espacement entre les éléments en longueur d'onde





## Fonctions 

#ADC
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
    
    for i in np.arange(0, len(signal_in_up)):
        if i == 0:
            signal_feedback[i] = signal_in_up[i]
        else:
            signal_feedback[i] = signal_in_up[i] - signal_out[i-1]

        if i == 0:
            signal_integrator[i] = signal_feedback[i]
        else:
            signal_integrator[i] = signal_feedback[i] + signal_integrator[i-1]

        signal_integrator[i] = Vpp / (2**enob) * np.round(signal_integrator[i] / (Vpp / (2**enob)))
    
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
    signal_out_decimate_centered = frequency_offset(signal_out_decimate, -f_tx, fs)
    return signal_out_decimate_centered


#Antenne
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

signal_in = frequency_offset(signal_qpsk_conv_noise,f_tx,fs) 
fft_qpsk_offset = np.fft.fft(signal_in)





## 2- Signal numérisé par l'ADC

signal_out = ADC_QPSK_GLOBAL(signal_in)
fft_out = np.fft.fft(signal_out)



SNR_phase = np.zeros(11)
SNR_time = np.zeros(11)



for du in np.arange(0, 1.1, 0.1): 
    
    dU = -du
    dV = du
    theta = np.arctan(np.sqrt((dU**2 + dV**2)))
    phi = np.arctan2(dU, dV) 
    theta_deg = np.rad2deg(theta)
    phi_deg = np.rad2deg(phi)
    

    
    ## 3- Décalage de phase pour les différents éléments à un angle fixé
    
    X,Y = generate_positions(N,d)
    phases = calculate_phases(X, Y, theta, phi)
    phases_factor = np.exp(1j * phases)
    number_of_wrapping_phase = np.floor(phases / (2*np.pi))
    max_wrapping = np.max(np.abs(number_of_wrapping_phase))
    
    signaux_phase = np.zeros((n_samples*samples_per_symbol, N, N), dtype=complex)
    
    for i in range(n_samples*samples_per_symbol):
        signaux_phase[i, :, :] = signal_out[i]
        signaux_phase[i, :, :] *= phases_factor
    
    
    
    ## 4- Décalage en temps pour les différents éléments à un angle fixé
    
    time_delay = calculate_time(phases)
    one_sample_time = 1/fs
    number_of_sample_time = np.round(time_delay/one_sample_time)
    max_sample_time = int(np.max(np.abs(number_of_sample_time)))
    
    signaux_time = np.zeros((n_samples*samples_per_symbol + 2*max_sample_time, N, N), dtype=complex)
    
    for i in range(n_samples*samples_per_symbol):
        signaux_time[i, :, :] = signal_out[i]
    
    for x in range(N):
        for y in range(N):
            signaux_time[:, x, y] = np.roll(signaux_time[:, x, y],max_sample_time + int(number_of_sample_time[x,y]))
       
    
    
    ## 5- GRD

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
    w = w * np.exp(1j * np.pi / lambda0 * (-X*dU  + Y*dV ))
    GRD_resultant = np.zeros((201, 201),dtype="complex")
    
    for i in range(N):
        for j in range(N):
            GRD_resultant = GRD_resultant + GRDS[i,j,:,:] * w[i,j]
    
    GRD_resultant_abs =np.abs(GRD_resultant)
    GRD_resultant_db = 20 * np.log10(GRD_resultant_abs)
    

    max_index = np.argmax(GRD_resultant_abs)
    k, l = np.unravel_index(max_index, GRD_resultant_abs.shape)

    
    
    
    
    ## 6- Somme des signaux en phase

    
    signaux_phase_grd = np.zeros((n_samples*samples_per_symbol, N, N), dtype=complex)
    fft_signaux_phase = np.zeros((n_samples*samples_per_symbol, N, N), dtype=complex)
    f = np.fft.fftfreq(n_samples*samples_per_symbol, d=1/fs)
    
    for i in range(N):
        for j in range(N):
            fft_signaux_phase[:,i,j] = np.fft.fft(signaux_phase[:,i,j])*np.exp(-2j * np.pi * f * one_sample_time * number_of_sample_time[i,j])
            signaux_phase_grd[:,i,j] = np.fft.ifft(fft_signaux_phase[:,i,j])
    
    for i in range(N):
        for j in range(N):
            signaux_phase_grd[:,i,j] = signaux_phase_grd[:,i,j]*np.abs(w[i,j])*GRDS[i,j,k,l]
    
    somme_signaux_phase = np.zeros(len(signaux_phase_grd))
    
    for i in range(N):
        for j in range(N):
            somme_signaux_phase = somme_signaux_phase +  signaux_phase_grd[:,i,j]
    
    hrrc = signal_qpsk[2]
    symbols = signal_qpsk[3]
    hrrc_inv = np.flipud(hrrc)
    demod_convolv_phase = signal.convolve(somme_signaux_phase, hrrc_inv,mode='same')
    symbols_out_demod_phase = demod_convolv_phase[0::samples_per_symbol]
    
    angle_diff = np.angle(symbols_out_demod_phase * np.conj(symbols))
    angle_moyen = np.mean(angle_diff)
    symbols_out_demod_egal_phase = symbols_out_demod_phase * np.exp(-1j * angle_moyen)
    
    plot_IQ_symbols(symbols_out_demod_egal_phase, title="Constellation de la somme des signaux en phase") 
    
    
    
    
    ## 7- Somme des signaux en temps
    
    signaux_time_grd = np.zeros((n_samples*samples_per_symbol + 2*max_sample_time, N, N), dtype=complex)
    
    for x in range(N):
        for y in range(N):
            signaux_time_grd[:, x, y] = np.roll(signaux_time[:, x, y],-max_sample_time - int(number_of_sample_time[x,y]))
    
    signaux_time_grd_trimmed = signaux_time_grd[:n_samples*samples_per_symbol, :, :]
    
    for i in range(N):
        for j in range(N):
            signaux_time_grd_trimmed[:,i,j] = signaux_time_grd_trimmed[:,i,j]*np.abs(w[i,j])*GRDS[i,j,k,l]
            
    somme_signaux_time = np.zeros(len(signaux_time_grd_trimmed))
    
    for i in range(N):
        for j in range(N):
            somme_signaux_time = somme_signaux_time + signaux_time_grd_trimmed[:,i,j]
            
    hrrc = signal_qpsk[2]
    symbols = signal_qpsk[3]
    hrrc_inv = np.flipud(hrrc)
    demod_convolv_time = signal.convolve(somme_signaux_time, hrrc_inv,mode='same')
    symbols_out_demod_time = demod_convolv_time[0::samples_per_symbol]
    
    angle_diff = np.angle(symbols_out_demod_time * np.conj(symbols))
    angle_moyen = np.mean(angle_diff)
    symbols_out_demod_egal_time = symbols_out_demod_time * np.exp(-1j * angle_moyen)
    
    plot_IQ_symbols(symbols_out_demod_egal_time, title="Constellation de la somme des signaux en temps") 




    SNR_phase[int(du*10)] = M2M4(symbols_out_demod_egal_phase,1)
    SNR_time[int(du*10)] = M2M4(symbols_out_demod_egal_time,1)

    print(f'{du} done')


x1 = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
x2 = np.array([0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1])


plt.plot(x2, SNR_phase, label='SNR phase', color='blue')
plt.plot(x2, SNR_time, label='SNR time', color='red')

plt.gca().invert_xaxis()

plt.title('SNR en fonction de dU et dV')
plt.xlabel('dU=-dV')
plt.ylabel('SNR')
plt.legend()
plt.show()





