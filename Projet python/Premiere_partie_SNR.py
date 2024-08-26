# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:21:23 2024

@author: IvanB
"""

from usual import *
from numba import *



## Paramètres 

Vpp = 2                     # Tension du DAC dans la boucle
enob = 4                    # Facteur de quantification
upsampling = 64             # Facteur de suréchantillonge
fs = 2*1000                 # Fréquence d'échantillonnage
time = 0.5                  # Durée du signal
n_samples = round(fs*time ) # Nombre d'échantillons
cutoff_freq = 200           # Fréquence de coupure du filtre pass-bas
samples_per_symbol = 100    # Nombre d'échantillons par symbole



## ADC delta sigma

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

@jit(nopython=True)
def ADC_Delta_Sigma(signal_in, Vpp, enob, upsampling):
    
    signal_in_up = ZOH(np.real(signal_in),upsampling)
    
    signal_feedback = np.zeros(len(signal_in_up))
    signal_integrator = np.zeros(len(signal_in_up))
    integrator_values = np.zeros(len(signal_in_up))
    signal_out = np.zeros(len(signal_in_up))
    bits_out = np.zeros(len(signal_in_up))
    quant_noise = np.zeros(len(signal_in_up))
    
    # Loop
    for i in np.arange(0,len(signal_in_up)):
        
        # Delta
        if i ==0:
            signal_feedback[i] = signal_in_up[i]
        if i>0:
            signal_feedback[i] = signal_in_up[i] - signal_out[i-1]

        
        # Intégrateur
        if i == 0:
            signal_integrator[i] =  signal_feedback[i]
        if i >0:
            signal_integrator[i] =  signal_feedback[i] + signal_integrator[i-1]

        signal_integrator[i] = Vpp/(2**enob)* np.round(signal_integrator[i]/(Vpp/(2**enob)))
    
        # Comparateur
        if signal_integrator[i] >= 0:
            signal_out[i] = Vpp/2
            bits_out[i] = 1
    
        if signal_integrator[i] < 0:
            signal_out[i] = -Vpp/2
            bits_out[i] = 0     
            
    return signal_out, bits_out



SNR = np.zeros(len(np.arange(0.1,0.5,0.01)))
Index = np.zeros(len(np.arange(0.1,0.5,0.01)))
k = 0

for j in np.arange(0.1,0.5,0.01):
    # Signaux initiaux
    noise = create_awgn(n_samples*samples_per_symbol, bandwidth=0.1)
    
    signal_qpsk = qpsk(n_samples,0.2,samples_per_symbol)
    signal_qpsk_conv = signal_qpsk[0]
    signal_qpsk_conv_noise = signal_qpsk_conv + noise
    fft_qpsk = np.fft.fft(signal_qpsk_conv_noise)
    
    signal_qpsk_offset = frequency_offset(signal_qpsk_conv_noise,fs*j,fs)
    signal_in = signal_qpsk_offset
    fft_in = np.fft.fft(signal_in) 
    
    # On rentre dans l'ADC
    signal_out, bits_out = ADC_Delta_Sigma(signal_in, Vpp, enob, upsampling)   
    signal_out_hilbert = signal.hilbert(signal_out)
    signal_out_filtered = lowpass_filter(signal_out_hilbert)
    signal_out_decimate = signal.decimate(signal_out_filtered,upsampling,ftype ='fir')
    signal_out_decimate_centered = lowpass_filter(frequency_offset(signal_out_decimate,-fs*j,fs))
    fft_out_decimate_centered = np.fft.fft(signal_out_decimate_centered)
    
    # Démodulation à la mano
    hrrc = signal_qpsk[2]
    symbols = signal_qpsk[3]
    hrrc_inv = np.flipud(hrrc)
    demod_convolv = signal.convolve(signal_out_decimate_centered, hrrc_inv,mode='same')
    symbols_out_demod = demod_convolv[0::samples_per_symbol]
    
    # Egalisation
    angle_diff = np.angle(symbols_out_demod * np.conj(symbols))
    angle_moyen = np.mean(angle_diff)
    symbols_out_demod_egal = symbols_out_demod * np.exp(-1j * angle_moyen)
    
    # Récupération du SNR
    SNR[k] =  M2M4(symbols_out_demod_egal,1)
    Index[k] = fs*j
    k = k+1



## Plot 

x = np.linspace(0, fs/2, len(SNR))

plt.plot(Index, SNR)
plt.title('SNR en fonction de la porteuse')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('SNR')

plt.show()




