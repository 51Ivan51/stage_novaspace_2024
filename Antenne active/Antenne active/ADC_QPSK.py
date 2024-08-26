# -*- coding: utf-8 -*-
from usual import *
from numba import *
import numba as nb

## Paramètres 

Vpp = 2                                                 # Tension du DAC dans la boucle
enob = 4                                                # Facteur de quantification
upsampling = 64                                         # Facteur de suréchantillonnage
fs = 60e9                                               # Fréquence d'échantillonnage (60GHz)
time = 0.0000002                                          # Durée du signal
n_samples = round(fs*time)                              # Nombre d'échantillons
cutoff_freq = 200                                       # Fréquence de coupure du filtre pass-bas
samples_per_symbol = 50                                 # Nombre d'échantillons par symbole


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
    signal_out = np.zeros(len(signal_in_up))
    
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
    
        if signal_integrator[i] < 0:
            signal_out[i] = -Vpp/2
            
    return signal_out




def ADC_QPSK_GLOBAL(signal_in):

     
    signal_out = ADC_Delta_Sigma(signal_in, Vpp, enob, upsampling) 
    signal_out_hilbert = signal.hilbert(signal_out)
    signal_out_filtered = lowpass_filter(signal_out_hilbert)
    signal_out_decimate = signal.decimate(signal_out_filtered,upsampling,ftype ='fir')
    signal_out_decimate_centered = frequency_offset(signal_out_decimate,-fs/10,fs)
    
   
    return signal_out_decimate_centered



## Signaux in

noise = create_awgn(n_samples*samples_per_symbol, bandwidth=0.1)
signal_qpsk = qpsk(n_samples,0.2,samples_per_symbol)
signal_qpsk_conv = signal_qpsk[0]
signal_qpsk_conv_noise = signal_qpsk_conv + noise
signal_in = frequency_offset(signal_qpsk_conv_noise,fs/10,fs)  
signal_out = ADC_QPSK_GLOBAL(signal_in)

hrrc = signal_qpsk[2]
symbols = signal_qpsk[3]
hrrc_inv = np.flipud(hrrc)
demod_convolv = signal.convolve(signal_out, hrrc_inv,mode='same')
symbols_out_demod = demod_convolv[0::samples_per_symbol]

angle_diff = np.angle(symbols_out_demod * np.conj(symbols))
angle_moyen = np.mean(angle_diff)
symbols_out_demod_egal = symbols_out_demod * np.exp(-1j * angle_moyen)

plot_IQ_symbols(symbols_out_demod_egal, title="Constellation démodulée + égalisation") 



print(f'Longueur de signal_out : {signal_out.shape}')
