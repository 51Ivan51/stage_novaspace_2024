# -*- coding: utf-8 -*-
from usual import *
from numba import *



## Paramètres 

Vpp = 2                         # Tension du DAC dans la boucle
enob = 3                        # Facteur de quantification
upsampling = 64                 # Facteur de suréchantillonge
fs = 2*1000                     # Fréquence d'échantillonnage
time = 0.5                      # Durée du signal
n_samples = round(fs*time )     # Nombre d'échantillons
cutoff_freq = 200               # Fréquence de coupure du filtre passe-bas



## Signaux in

t = np.linspace(0, time, n_samples)
sinus = np.sin(2 * np.pi * 50 * t)
fft_sin = np.fft.fft(sinus)

signal_in = sinus
fft_in = np.fft.fft(signal_in) 



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


signal_out, bits_out = ADC_Delta_Sigma(signal_in, Vpp, enob, upsampling)   
fft_out = np.fft.fft(signal_out)

signal_out_filtered = lowpass_filter(signal_out)
fft_out_filtered = np.fft.fft(signal_out_filtered)

signal_out_decimate = signal.decimate(signal_out_filtered,upsampling,ftype ='fir')
fft_out_decimate = np.fft.fft(signal_out_decimate)



## Plot 

plt.figure(figsize=(12, 8))
x = np.linspace(0, time, n_samples)
freq = np.fft.fftfreq(n_samples, 1/fs)
freq_fs = np.fft.fftfreq(n_samples*upsampling, 1/fs)


plt.subplot(5, 1, 1)
plt.plot(x, signal_in)
plt.title('Signal in')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')

plt.subplot(5, 1, 2)
plt.plot(x, signal_out_decimate)
plt.title('Signal out filtered + decimated')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')

plt.subplot(5, 1, 3)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq, (1/fs)*np.abs(fft_in))
plt.title('FFT in')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')

plt.subplot(5, 1, 4)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq_fs, (1/fs)*np.abs(fft_out))
plt.title('FFT out (en sortie de l ADC)')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')

plt.subplot(5, 1, 5)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq, (1/fs)*np.abs(fft_out_decimate))
plt.title('FFT out filtered + decimated')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')


plt.tight_layout()
plt.show()