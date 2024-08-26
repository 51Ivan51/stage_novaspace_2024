# -*- coding: utf-8 -*-
from usual import *
from numba import *
import numba as nb

## Paramètres 
Vpp = 2                     # Tension du DAC dans la boucle
enob = 3                    # Facteur de quantification
upsampling = 64             # Facteur de suréchantillonge
fs = 2*1000                 # Fréquence d'échantillonnage
time = 0.5                  # Durée du signal
n_samples = round(fs*time ) # Nombre d'échantillons
cutoff_freq = 200           # Fréquence de coupure du filtre pass-bas
samples_per_symbol = 20     # Nombre d'échantillons par symbole



## Signaux in

# On cree le bruit
noise = create_awgn(n_samples*samples_per_symbol, bandwidth=0.1)
fft_noise = np.fft.fft(noise)

# On cree le signal QPSK
signal_qpsk = qpsk(n_samples,0.2,samples_per_symbol)
signal_qpsk_conv = signal_qpsk[0]
signal_qpsk_conv_noise = signal_qpsk_conv + noise
fft_qpsk = np.fft.fft(signal_qpsk_conv_noise)

# On déplace le signal avec une porteuse
signal_qpsk_offset = frequency_offset(signal_qpsk_conv_noise,fs/10,fs)  
fft_qpsk_offset = np.fft.fft(signal_qpsk_offset)

# Le signal qui va rentrer dans l'ADC
signal_in = signal_qpsk_offset
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

# Récupération du signal en sortie de l'ADC
signal_out, bits_out = ADC_Delta_Sigma(signal_in, Vpp, enob, upsampling)   
fft_out = np.fft.fft(signal_out)

# Transformée de Hilbert pour récuperer les symboles complexes
signal_out_hilbert = signal.hilbert(signal_out)
fft_hilbert = np.fft.fft(signal_out_hilbert)

# Filtrer le signal par un passe bas
signal_out_filtered = lowpass_filter(signal_out_hilbert)
fft_out_filtered = np.fft.fft(signal_out_filtered)

# Décimation du signal
signal_out_decimate = signal.decimate(signal_out_filtered,upsampling,ftype ='fir')
fft_out_decimate = np.fft.fft(signal_out_decimate)

# Revenir en bande de base
signal_out_decimate_centered = frequency_offset(signal_out_decimate,-fs/10,fs)
fft_out_decimate_centered = np.fft.fft(signal_out_decimate_centered)


# Démodulation à la main
hrrc = signal_qpsk[2]
symbols = signal_qpsk[3]
hrrc_inv = np.flipud(hrrc)
demod_convolv = signal.convolve(signal_out_decimate_centered, hrrc_inv,mode='same')

symbols_out_demod = demod_convolv[0::samples_per_symbol]
fft_out_demod_mano = np.fft.fft(symbols_out_demod)

# Egalisation
angle_diff = np.angle(symbols_out_demod * np.conj(symbols))
angle_moyen = np.mean(angle_diff)
symbols_out_demod_egal = symbols_out_demod * np.exp(-1j * angle_moyen)



## Plot 

# Constellations
plot_IQ_symbols(signal_qpsk[3],title="Constellation initiale")                         # Signal in
plot_IQ_symbols(symbols_out_demod, title="Constellation démodulée ")                   # Signal démodulé
plot_IQ_symbols(symbols_out_demod_egal, title="Constellation démodulée + égalisation") # Signal démodulé avec égalisation



# Signaux temporels et fft

plt.figure(figsize=(12, 8))
x = np.linspace(0, time, n_samples*samples_per_symbol)
x_fs = np.linspace(0, time, n_samples*upsampling*samples_per_symbol)
x_demod =  np.linspace(0, time, n_samples)
freq = np.fft.fftfreq(n_samples*samples_per_symbol, 1/fs)
freq_fs = np.fft.fftfreq(n_samples*upsampling*samples_per_symbol, 1/fs)
freq_demod = np.fft.fftfreq(n_samples, 1/fs)


plt.subplot(5, 1, 1)
plt.plot(x, signal_in)
plt.title('Signal in')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')


plt.subplot(5, 1, 2)
plt.plot(x, signal_out_decimate_centered)
plt.title('Signal out filtered + decimated + centered')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')


plt.subplot(5, 1, 3)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq, (1/fs)*np.abs(fft_qpsk))
plt.title('FFT in')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')

plt.subplot(5, 1, 4)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq, (1/fs)*np.abs(fft_out_decimate))
plt.title('FFT out filtered + decimated')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')

plt.subplot(5, 1, 5)
plt.xlim(-fs/2-50,fs/2+50)
plt.plot(freq, (1/fs)*np.abs(fft_out_decimate_centered))
plt.title('FFT out filtered + decimated + centered')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')



plt.tight_layout()
plt.show()
