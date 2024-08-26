# -*- coding: utf-8 -*-
"""

@author: nicolas JACQUEY

Usual functions for signal processing

"""

# Usual import
import numpy as np
from scipy import signal
from bokeh.plotting import figure, show,curdoc
from bokeh.models import BoxAnnotation
import scipy
import matplotlib.pyplot as plt

# Importing GPS signal library
import gnsstools.gps.l1cd as l1
import gnsstools.gps.l5i as l5i
import gnsstools.gps.l5q as l5q
from numpy import matlib as mp

def amplifier(signal_in, hpa):
    """
    

    Parameters
    ----------
    signal_in : numpy array
        Input signal
    hpa : string
        link to the AMAM/AMPM model file
        The file shall be in the following format
        HPA AMAM AMPM model. Column 1 = IBO, Column 2 = OBO, Column 3 = Phase shift

    Returns
    -------
    signal_out : numpy array
        Complex  signal after amplification

    """
    
    ibo_range = hpa[:,0]
    obo_range = hpa[:,1]
    phase_range = hpa[:,2]
    
    signal_in_am = 20*np.log10(np.abs(signal_in))
    signal_in_pm = np.angle(signal_in)
    signal_out_am = np.interp(signal_in_am,ibo_range, obo_range)
    signal_out_pm = np.interp(signal_in_am,ibo_range, phase_range)*np.pi/180

    signal_out = 10**(signal_out_am/20)*np.exp(1j*(signal_in_pm+signal_out_pm))
    
    return signal_out

""" Mean power of a signal """
def power_mean(signal_in):
    """
    
    Parameters
    ----------
    signal_in : numpy array
        input signal.

    Returns
    -------
    float
        mean power of the input signal.

    """
    return np.mean(np.abs(signal_in)**2)

""" Cross correlation function """
def xcorr(x, y, scale='none'):
    """
    r = xcorr(x,y) returns the cross-correlation of two discrete-time sequences. Cross-correlation measures the similarity between a vector x and shifted (lagged) copies of a vector y as a function of the lag. If x and y have different lengths, the function appends zeros to the end of the shorter vector so it has the same length as the other.

    Parameters
    ----------
    x : numpy complex or real array
        DESCRIPTION.
    y : numpy complex or real array
        DESCRIPTION.
    scale : string, OPTIONAL
        DESCRIPTION. The default is 'none'.

    Returns
    -------
    corr : numpy array
        cross-correlation of two discrete-time sequences

    """
    
    
    # Pad shorter array if signals are different lengths
    if x.size > y.size:
        pad_amount = x.size - y.size
        y = np.append(y, np.repeat(0, pad_amount))
    elif y.size > x.size:
        pad_amount = y.size - x.size
        x = np.append(x, np.repeat(0, pad_amount))

    corr = np.correlate(x, y, mode='full')  # scale = 'none'
    lags = np.arange(-(x.size - 1), x.size)

    if scale == 'biased':
        corr = corr / x.size
    elif scale == 'unbiased':
        corr /= (x.size - abs(lags))
    elif scale == 'coeff':
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))
        
    return corr

""" C/I function. Provide the IM signal """
def CoverI(signal1,signal2): #Put the demodulated symbols of each signal
    LS1 = signal1.size
    LS2 = signal2.size
    #Correlation = np.correlate(signal1,signal2,'full')
    Correlation = xcorr(signal1,signal2,'biased')
    Power_Tx = power_mean(signal1)
    PeakCorr = np.argwhere(np.abs(Correlation)== np.max(np.abs(Correlation)))
    Gain =np.conj(Correlation[int(PeakCorr)])/Power_Tx
    Delay = np.abs(signal1.size-int(PeakCorr))-1
    Signal_Interferences = signal2[int(Delay):LS2] - Gain* signal1[0:LS1-int(Delay)]
    Power_Rx = power_mean(Gain*signal1)
    Power_Interf = power_mean(Signal_Interferences)
    C_I = 10*np.log10(Power_Rx/Power_Interf)

    return C_I,Gain, Delay, Signal_Interferences


""" GPS signal sampling function """
def sample(code,nsamples,chipRate,samplFreq,sampPhase):
    len_code = len(code)   
    idxs = np.floor(np.arange(0,nsamples-1) * (chipRate/samplFreq) + sampPhase)
    idxs = 1+np.mod(idxs, len_code)
    idxs=idxs.astype(int)
    return  code[idxs]

""" Root Raised Cosine Filter
input
    syms = define impulse filter length, typically equal to 10. Filter delay is equal to syms * P
    beta = rolloff
    P = number of samples per symbol
output
    filter response
"""
def root_raised_cos(syms,beta,P):
    k= np.arange(-syms*P+1e-8,(syms*P+1e-8)+1)
    s = (np.sin(np.pi*(1-beta)*k/P)+4*beta*k/P*np.cos(np.pi*k/P*(1+beta)))/(np.pi*k/P*(1-16*(beta**2)*(k/P)**2))
    s=s/np.sqrt(np.sum(np.abs(s)**2))
    return s

""" upsampling of signal by factor N """    
def upsample(signal,N):
    y = [0] * (N * len(signal))
    y[::N] = signal
    y=np.array(y)
    return y

""" downsampling of signal by factor N """
def downsample(signal,N):
    """
    

    Parameters
    ----------
    signal : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return signal[::N]

def CW(nsamples,frequency, frequency_sampling):
    """
    

    Parameters
    ----------
    nsamples : TYPE
        DESCRIPTION.
    frequency : TYPE
        DESCRIPTION.
    frequency_sampling : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    signal_in = np.ones(nsamples)
    return frequency_offset(signal_in,frequency,frequency_sampling)


""" Creation of QPSK signal """
def qpsk(nb_symbols,beta,P):
    """
    

    Parameters
    ----------
    nb_symbols : int
        number of symbols to be generated
    beta : TYPE
        rolloff
    P : TYPE
        number of samples per symbol

    Returns
    -------
    TYPE
        DESCRIPTION.
    delay : TYPE
        DESCRIPTION.
    h_rrc : TYPE
        DESCRIPTION.

    """
    h_rrc = root_raised_cos(10,beta,P)
    symbols =np.random.randint(4, size=nb_symbols)
    signal_raw = upsample(np.exp(1j*np.pi*(symbols/2+1/4)),P)
    delay = (len(h_rrc)-1)/2
    symbols = np.exp(1j*np.pi*(symbols/2+1/4))
    signal_qpsk = signal.convolve(signal_raw, h_rrc,mode='same')
    return signal_qpsk, delay, h_rrc, symbols

""" Creation of BPSK signal """
def bpsk(nb_symbols,beta,P):
    """
    

    Parameters
    ----------
    nb_symbols : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    delay : TYPE
        DESCRIPTION.
    h_rrc : TYPE
        DESCRIPTION.

    """
    h_rrc = root_raised_cos(10,beta,P)
    symbols =np.random.randint(2, size=nb_symbols)
    signal_raw = upsample(np.exp(1j*np.pi*symbols),P)
    delay = (len(h_rrc)-1)/2
    symbols = np.exp(1j*np.pi*symbols)
    signal_bpsk = signal.convolve(signal_raw, h_rrc,mode='same')
    return signal_bpsk, delay, h_rrc, symbols

""" Creation of 8PSK signal """
def psk8(nb_symbols,beta,P):
    """
    

    Parameters
    ----------
    nb_symbols : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    delay : TYPE
        DESCRIPTION.
    h_rrc : TYPE
        DESCRIPTION.

    """
    h_rrc = root_raised_cos(10,beta,P)
    symbols =np.random.randint(8, size=nb_symbols)
    signal_raw = upsample(np.exp(1j*np.pi*(symbols/4+1/8)),P)
    delay = (len(h_rrc)-1)/2
    symbols = np.exp(1j*np.pi*(symbols/4+1/8))
    signal_psk8 = signal.convolve(signal_raw, h_rrc,mode='same')
    return signal_psk8 , delay, h_rrc,symbols
    
""" Creation of 16APSK signal """
def apsk16(nb_symbols,beta,P):
    """
    

    Parameters
    ----------
    nb_symbols : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    delay : TYPE
        DESCRIPTION.
    h_rrc : TYPE
        DESCRIPTION.

    """
    h_rrc = root_raised_cos(10,beta,P)
    symbols =np.random.randint(16, size=nb_symbols)
    index_1st = np.where(symbols <=3)
    index_2nd = np.where(symbols >3)
    symbols_raw = symbols.astype(complex) *0
    symbols_raw[index_1st] = symbols.astype(complex)[index_1st]/2+1/4
    symbols_raw[index_2nd] = symbols.astype(complex)[index_2nd]/6
    
    signal_raw= symbols_raw *0
    signal_raw[index_1st] = np.exp(1j*np.pi*symbols_raw[index_1st])
    signal_raw[index_2nd] = 2.21 * np.exp(1j*np.pi*symbols_raw[index_2nd])
    
    symbols = signal_raw.copy()
    
    signal_raw = upsample(signal_raw, P)
    delay = (len(h_rrc)-1)/2
    signal_apsk16 = signal.convolve(signal_raw, h_rrc,mode='same')
    return signal_apsk16, delay, h_rrc,symbols

""" Creation of 32APSK signal """

def apsk32(nb_symbols,beta,P):
    """
    

    Parameters
    ----------
    nb_symbols : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    delay : TYPE
        DESCRIPTION.
    h_rrc : TYPE
        DESCRIPTION.

    """
    h_rrc = root_raised_cos(10,beta,P)
    symbols =np.random.randint(32, size=nb_symbols)
    index_1st = np.where(symbols <=3)
    index_2nd = np.where(np.logical_and(symbols >3,symbols <=15))
    index_3rd = np.where(symbols>15)
    symbols_raw = symbols.astype(complex) *0
    symbols_raw[index_1st] = symbols.astype(complex)[index_1st]/2
    symbols_raw[index_2nd] = symbols.astype(complex)[index_2nd]/6
    symbols_raw[index_3rd] = symbols.astype(complex)[index_3rd]/8
    
    signal_raw= symbols_raw *0
    signal_raw[index_1st] = np.exp(1j*np.pi*symbols_raw[index_1st])
    signal_raw[index_2nd] = 2.54 * np.exp(1j*np.pi*symbols_raw[index_2nd])
    signal_raw[index_3rd] = 4.23 * np.exp(1J*np.pi*symbols_raw[index_3rd])
    symbols = signal_raw.copy()
    
    signal_raw = upsample(signal_raw, P)
    delay = (len(h_rrc)-1)/2
    signal_apsk32 = signal.convolve(signal_raw, h_rrc,mode='same')
    return signal_apsk32, delay, h_rrc,symbols


""" Creation of L1 SBAS Signal """
def L1_Signal(nb_symbols, Fs, Fc):
    """
    

    Parameters
    ----------
    nb_symbols : TYPE
        DESCRIPTION.
    Fs : TYPE
        DESCRIPTION.
    Fc : TYPE
        DESCRIPTION.

    Returns
    -------
    signal_L1 : TYPE
        DESCRIPTION.
    coh_time : TYPE
        DESCRIPTION.

    """
    Puissance_L1=10**(16.2/10)
    # I/Q creation
    # I created first
    x=2*(l1.l1cd(12)-0.5)
    n_periods = 0.1;
    coh_time = n_periods*1e-3;
    signalI = sample(x, nsamples = Fs*coh_time, chipRate = 1.023e6, samplFreq = Fs, sampPhase = 0)
    # then Q
    signalI_temp = signalI.copy() # True copy
    signalQ = 0*signalI_temp
    # full L1 signal baseband
    signal_L1 = signalI + 1j*signalQ
    Fc=0;
    i=np.array(0+1j) #imaginary number
    carrier_L1 = np.exp(2*np.pi*np.arange(0,len(signal_L1))*i*Fc/Fs)
    # to the L1 frequency
    signal_L1 = carrier_L1 * signal_L1 * np.sqrt(Puissance_L1)
    signal_L1 = signal_L1 / power_mean(signal_L1)**(1/2)
    
    return signal_L1, coh_time

""" Creation of L1 SBAS Signal """
def L5_Signal(nb_symbols, Fs, Fc):
    """
    

    Parameters
    ----------
    nb_symbols : TYPE
        DESCRIPTION.
    Fs : TYPE
        DESCRIPTION.
    Fc : TYPE
        DESCRIPTION.

    Returns
    -------
    signal_L5 : TYPE
        DESCRIPTION.
    coh_time : TYPE
        DESCRIPTION.

    """
    Puissance_L5=10**(16.2/10)
    # I/Q creation
    # I created first
    x=2*(l5i.l5i_code(12)-0.5)
    n_periods = 0.1;
    coh_time = n_periods*1e-3;
    signalI = sample(x, nsamples = Fs*coh_time, chipRate = 10*1.023e6, samplFreq = Fs, sampPhase = 0)
    # then Q
    x=2*(l5q.l5q_code(12)-0.5)
    signalQ = sample(x, nsamples = Fs*coh_time, chipRate = 10*1.023e6, samplFreq = Fs, sampPhase = 0)
    Fc=0;
    signal_L5 = signalI + 1j*signalQ
    i=np.array(0+1j) #imaginary number
    carrier_L5 = np.exp(2*np.pi*np.arange(0,len(signal_L5))*i*Fc/Fs)
    # to the L5 frequency
    signal_L5 = carrier_L5 * signal_L5 * np.sqrt(Puissance_L5)
    signal_L5 = signal_L5 / power_mean(signal_L5)**(1/2)

    return signal_L5, coh_time

""" Applying Frequency offset to a signal """
def frequency_offset(signal_in,frequency,Fs):
    """
    

    Parameters
    ----------
    signal_in : TYPE
        DESCRIPTION.
    frequency : TYPE
        DESCRIPTION.
    Fs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    time = np.arange(0,signal_in.size)
    signal_LO = np.exp(1j*2*np.pi*frequency*time/Fs)
    return signal_in*signal_LO

""" signal normalization """
def normalize(signal_in):
    """
    

    Parameters
    ----------
    signal_in : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return signal_in / power_mean(signal_in)**(1/2)

"""
M2M4 SNR estimator
"""
def M2M4(RxSymbolsWithoutPilots,signal_kurtosis):
    """
    

    Parameters
    ----------
    RxSymbolsWithoutPilots : TYPE
        DESCRIPTION.
    signal_kurtosis : TYPE
        DESCRIPTION.

    Returns
    -------
    EstimatedSNR : TYPE
        DESCRIPTION.

    """
    M2 = np.mean(np.abs(RxSymbolsWithoutPilots)**2)
    M4 = np.mean(np.abs(RxSymbolsWithoutPilots)**4)
    noise_kurtosis = 2
    tmp = (4-signal_kurtosis*noise_kurtosis)*M2*M2 + M4*(signal_kurtosis+noise_kurtosis-4)
    tmp = np.sign(tmp)*tmp
    Se = np.sqrt(tmp)/(signal_kurtosis-2)
    Se = np.sign(Se)*Se
    Ne= np.abs(M2-Se)
    EstimatedSNR = 10*np.log10(Se/Ne)
    return EstimatedSNR

""" Creat I/Q constellation plot"""
def plot_IQ(signal_in, samples_per_symbol, h_rccs, remove_edges = False, title = 'I/Q Constellation'):
    """
    

    Parameters
    ----------
    signal_in : TYPE
        DESCRIPTION.
    samples_per_symbol : TYPE
        DESCRIPTION.
    h_rccs : TYPE
        DESCRIPTION.
    remove_edges : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is 'I/Q Constellation'.

    Returns
    -------
    None.

    """
    signal_temp = signal_in.copy()
    if type(h_rccs) != float or type(h_rccs) != int:
        h_rccs_temp = h_rccs.copy()    
        signal_temp = np.convolve(signal_temp, h_rccs_temp, 'full')
    # Extract symbols
    symbols = signal_temp[::samples_per_symbol]
    if remove_edges == True:
        symbols = symbols[100:len(symbols) - 100]
    # Separate real and imaginary parts
    real_part = np.real(symbols)
    imaginary_part = np.imag(symbols)
    # Create I/Q constellation plot
    plt.figure(figsize=(6, 6))
    plt.scatter(real_part, imaginary_part, s=5, marker='o', color='b')
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title(title)
    plt.grid()
    plt.axis('equal')  # Ensure the axis scales are equal for a correct representation of the constellation
    plt.show()

def plot_IQ_symbols(symbols_in, remove_edges = False, title = 'I/Q Constellation'):
    """
    

    Parameters
    ----------
    symbols_in : TYPE
        DESCRIPTION.
    remove_edges : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is 'I/Q Constellation'.

    Returns
    -------
    None.

    """
    symbols_in_copy = symbols_in.copy()
    
    if remove_edges == True:
        symbols_in = symbols_in[10000:len(symbols_in) - 10000]
    # Separate real and imaginary parts
    real_part = np.real(symbols_in)
    imaginary_part = np.imag(symbols_in)
    # Create I/Q constellation plot
    plt.figure(figsize=(6, 6))
    plt.scatter(real_part, imaginary_part, s=5, marker='o', color='b')
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title(title)
    plt.grid()
    plt.axis('equal')  # Ensure the axis scales are equal for a correct representation of the constellation
    plt.show()
    
    symbols_in = symbols_in_copy.copy()
    
# Quantizer
def quantizer(u,q):
    """
    

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return q * np.round(u/q)

# Standard ADC
def ADC(signal_in,sampling_freq_in,sampling_freq_out,enob,Vpp):
    """
    

    Parameters
    ----------
    signal_in : TYPE
        DESCRIPTION.
    sampling_freq_in : TYPE
        DESCRIPTION.
    sampling_freq_out : TYPE
        DESCRIPTION.
    enob : TYPE
        DESCRIPTION.
    Vpp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    signal_temp = downsample(signal_in,int(sampling_freq_in/sampling_freq_out))
    signal_temp = np.real(signal_temp)
    signal_temp = np.clip(signal_temp,-Vpp/2,Vpp/2)
    signal_temp = quantizer(signal_temp,Vpp/(2**enob))

    return signal.hilbert(signal_temp)

# Standard DAC
def DAC(signal_in,sampling_freq_in,sampling_freq_out,enob,Vpp):
    """
    

    Parameters
    ----------
    signal_in : TYPE
        DESCRIPTION.
    sampling_freq_in : TYPE
        DESCRIPTION.
    sampling_freq_out : TYPE
        DESCRIPTION.
    enob : TYPE
        DESCRIPTION.
    Vpp : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    #signal_temp = upsample(signal_in,int(sampling_freq_out/sampling_freq_in))
    signal_temp = ZOH(signal_in,int(sampling_freq_out/sampling_freq_in))
    signal_temp = np.real(signal_temp)
    signal_temp = np.clip(signal_temp,-Vpp/2,Vpp/2)
    signal_temp = quantizer(signal_temp,Vpp/(2**enob))

    return signal.hilbert(signal_temp)    


# EVM - DEBUG
def calculate_evm(reference_signal, measured_signal, N, samples_per_symbol):
    """
    

    Parameters
    ----------
    reference_signal : TYPE
        DESCRIPTION.
    measured_signal : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    samples_per_symbol : TYPE
        DESCRIPTION.

    Returns
    -------
    evm_db : TYPE
        DESCRIPTION.

    """
    # Ensure both signals are numpy arrays
    reference_signal = np.array(reference_signal)
    measured_signal = np.array(measured_signal)

    # Remove the first and last N symbols from both signals
    reference_signal_trimmed = reference_signal[N:-N]
    measured_signal_trimmed = measured_signal[N:-N]

    # Get the symbols
    reference_symbols  = np.real(reference_signal_trimmed[::samples_per_symbol]) + 1j * np.imag(reference_signal_trimmed[::samples_per_symbol])
    measured_symbols  = np.real(measured_signal_trimmed[::samples_per_symbol]) + 1j * np.imag(measured_signal_trimmed[::samples_per_symbol])
    
    # Perform cross-correlation to find optimal phase offset
    cross_corr = np.correlate(reference_symbols.flatten(), measured_symbols.flatten(), mode='full')
    phase_offset_index = np.argmax(np.abs(cross_corr)) - (len(reference_symbols) - 1)
    phase_offset_radians = 2 * np.pi * phase_offset_index / len(reference_symbols)

    # Rotate the measured symbols to align with the reference signal in phase
    measured_symbols_aligned = measured_symbols * np.exp(-1j * phase_offset_radians)

    # Calculate the absolute difference between the aligned measured and reference symbols
    error_vector = np.abs(measured_symbols_aligned - reference_symbols)

    # Calculate the RMS of the absolute error vector
    rms_error = np.sqrt(np.mean(np.square(error_vector)))

    # Calculate the RMS of the absolute reference symbols
    rms_reference = np.sqrt(np.mean(np.square(np.abs(reference_symbols))))

    # Calculate the EVM in dB
    evm_db = 20 * np.log10(rms_error / rms_reference)

    return evm_db

def pwelch(signal_in,Fs =1, title ='Title', x_axis_label="Frequency (Hz)", y_axis_label="Power Spectral Density (dBW/Hz)"):
    """
    

    Parameters
    ----------
    signal_in : TYPE
        DESCRIPTION.
    Fs : TYPE, optional
        DESCRIPTION. The default is 1.
    title : TYPE, optional
        DESCRIPTION. The default is 'Title'.
    x_axis_label : TYPE, optional
        DESCRIPTION. The default is "Frequency (Hz)".
    y_axis_label : TYPE, optional
        DESCRIPTION. The default is "Power Spectral Density (dBW/Hz)".

    Returns
    -------
    f11 : TYPE
        DESCRIPTION.
    Pxx_den1 : TYPE
        DESCRIPTION.

    """
    f1, Pxx_den1 = signal.welch(signal_in, Fs, nperseg=2056)
    #f1, Pxx_den1 = signal.periodogram(signal_in, Fs)
    index1 = np.argsort(f1)
    f11=f1[index1]
    Pxx_den1 = Pxx_den1[index1]
    
    # Filter only positive frequencies
    positive_freq_mask = f11 >= 0
    f11 = f11[positive_freq_mask]
    Pxx_den1 = Pxx_den1[positive_freq_mask]


    p1 = figure(title=title,x_axis_label = x_axis_label, y_axis_label = y_axis_label)
    # p1.line(f1, 10*np.log10(Pxx_den_out1),legend_label = "L1 SSPA output",color="red")
    p1.line(f11, 10*np.log10(Pxx_den1),legend_label = title,color="green")
    show(p1)
    return f11, Pxx_den1
    

def create_awgn(nsamples,frequency=0, bandwidth=0.25, frequency_sampling=1,std_dev=0.01):
    """
    Create complex signal of limited bandwidth to avoid negative frequenct replica after filtering
    
    Parameters
    ----------
    nsamples : double or integer
        number of samples in the returned signal
    frequency : TYPE, optional
        Frequency centre of the noise. The default is 0.
    bandwidth : TYPE, optional
        Bandwidth of the noise. The default is 0.25.
    frequency_sampling : TYPE, optional
        Frequency sampling. The default is 1.
    std_dev : TYPE, optional
        Standard deviation. The default is 0.01.

    Returns
    -------
    Complex array of length nsamples representing a white noise with a limited bandwidth and 
    frequency shifted
    
        
    """

    b, a = signal.ellip(10, 0.01, 120, bandwidth/frequency_sampling)
    signal_in = np.random.normal(0, std_dev, nsamples) + 1j * np.random.normal(0, std_dev, nsamples)
    return frequency_offset(signal.filtfilt(b,a, signal_in),frequency,frequency_sampling)


def add_awgn_to_complex_signal(signal, std_dev = 0.1):
    """
    Add Additive White Gaussian Noise (AWGN) to a complex signal.

    Parameters:
        signal (numpy.array): The input complex signal.
        std_dev (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.array: The complex signal with AWGN added.
    """
    # Mean of the Gaussian noise (usually set to 0 for AWGN)
    mean = 0

    # Number of samples (same as the length of the complex signal)
    num_samples = len(signal)

    # Generate AWGN samples
    awgn_samples = np.random.normal(mean, std_dev, num_samples) + 1j * np.random.normal(mean, std_dev, num_samples)

    # Add the generated AWGN samples to the complex signal
    noisy_signal = signal + awgn_samples

    return noisy_signal  
  
    
def add_phase_noise(signal_in,Fs,phase_noise_freq,phase_noise_power,VALIDATION_ON):
    """
    

    Parameters
    ----------
    signal_in : numpy complex array
        input COMPLEX signal
    Fs : float
        frequency sampling.
    phase_noise_freq : numpy array
        frequencies at which SSB Phase Noise is defined (offset from carrier in Hz)
    phase_noise_power : numpy array
        SSB Phase Noise power ( in dBc/Hz )
    VALIDATION_ON : boolean
        DESCRIPTION.

    Returns
    -------
    signal_out : numpy complex array
        signal with added phase noise

    """
    realmin = np.finfo(float).tiny

    # sort phase noise freq and phase nois power
    index_freq = np.argsort(phase_noise_freq)
    phase_noise_freq = np.sort(phase_noise_freq)
    phase_noise_power = phase_noise_power[index_freq]

    # Add 0 dBc/Hz @ DC
    phase_noise_power = np.insert(phase_noise_power,0,0)
    phase_noise_freq = np.insert(phase_noise_freq,0,0)

    n_samples = len(signal_in)
    if np.remainder(n_samples,2):
      M = (n_samples + 1) / 2 + 1
    else:
      M = n_samples / 2 + 1


    #Equally spaced partitioning of the half spectrum
    F  = np.linspace( 0, Fs/2, int(M) )
    dF =np.diff(F)
    dF= np.append(dF, F[-1]-F[-2])

    #Perform interpolation of phase_noise_power in log-scale
    intrvlNum = len(phase_noise_freq)
    logP = np.zeros(int(M))

    for intrvlIndex in np.arange(0,intrvlNum):
      leftBound = phase_noise_freq[intrvlIndex]
      t1 = phase_noise_power[intrvlIndex]

      if intrvlIndex == intrvlNum-1:
          rightBound = Fs / 2
          t2 = phase_noise_power[-1]
          inside = np.where( (F >= leftBound) & (F <= rightBound))
      else:
          rightBound = phase_noise_freq[intrvlIndex + 1]
          t2 = phase_noise_power[intrvlIndex + 1]
          inside = np.where( (F >= leftBound)  & (F < rightBound) )

      print(inside)
      logP[inside] = t1 + (np.log10(F[inside] + realmin) - np.log10(leftBound + realmin)) / (np.log10(rightBound + realmin) - np.log10(leftBound + realmin)) * (t2 - t1)

    P = 10.0 ** (np.real(logP) / 10)

    if not VALIDATION_ON :
      awgn_P1 = (np.sqrt(0.5) * (np.random.randn(int(M)) + 1j * np.random.randn(int(M))))
    else:
      awgn_P1 = (np.sqrt(0.5) * (np.ones((int(M))) + 1j * np.ones((int(M)))))

    # Shape the noise on the positive spectrum [0, Fs/2] including bounds ( M points )
    X = (2 * M - 2) * np.sqrt(dF * P) *awgn_P1

    # Complete symmetrical negative spectrum  (Fs/2, Fs) not including bounds (M-2 points)
    #X[M -1 + np.arange(1,M - 2)] = np.fliplr(np.conjugate(X[1:-2]))
    X=np.append(X,  np.flipud(np.conjugate(X[1:-1])))
    X[0]=0 #Remove DC

    x=np.fft.ifft(X)
    phase_noise = np.exp(1j * np.real(x))


    if not VALIDATION_ON :
      signal_out = signal_in * phase_noise
    else:
      signal_out = signal_in * phase_noise

    return signal_out



def rref(F, K, M):
    x = K*(2*M*F-0.5);
    A = np.sqrt(0.5*scipy.special.erfc(x))
    return A

def npr_coeff(N = 256, L = 128, K = 8):
    """
    %NPR_COEFF generates near NPR filter bank coefficients.
    %  COEFF = NPR_COEFF(N,L) generates the filter coefficients
    %  for a near perfect reconstruction filter bank.
    %  The number of channels will be N, and L is the number of 
    %  filter taps used per channel. The output COEFF will have 
    %  size (N/2,L).
    %
    %  The prototype is constructed starting with an equiripple
    %  approximation to a 'root raised error function' shaped filter.
    %
    %  NPR_COEFF(N,L) with no output arguments plots the magnitude
    %  and the prototype filter in the current figure window.
    %
    %  See also npr_analysis, npr_synthesis, npr_coeff.
    %
    % (c) 2007 Wessel Lubberhuizen

    Parameters
    ----------
    N : TYPE, optional
        DESCRIPTION. The default is 256.
    L : TYPE, optional
        DESCRIPTION. The default is 128.
    K : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------
    coeff : TYPE
        DESCRIPTION.

    """
    if L == 8: K = 4.853;
    elif L == 10: K = 4.775;
    elif L == 12: K = 5.257;
    elif L == 14: K = 5.736;
    elif L == 16: K = 5.856;
    elif L == 18: K = 7.037;
    elif L == 20: K = 6.499;
    elif L == 22: K = 6.483;
    elif L == 24: K = 7.410;
    elif L == 26: K = 7.022;
    elif L == 28: K = 7.097;
    elif L == 30: K = 7.755;
    elif L == 32: K = 7.452;
    elif L == 48: K = 8.522;
    elif L == 64: K = 9.457;
    elif L == 96: K = 10.785;
    elif L == 128: K = 11.5;
    elif L == 192: K = 11.5;
    elif L == 256: K = 11.5;
    else: K = 8
    
    N=float(N)
    M = N/2;
    F = np.linspace(start = 0, stop = L*M-1, num = int(L*M))/(L*M);
    F = np.reshape(F, (1, int(L*M)))

    A = rref(F, K, M);
    N = len(A[0]);

    window_range_1 = N - np.linspace(start = 0, stop = N/2-1, num = int(N/2));
    window_range_2 = 2 + np.linspace(start = 0, stop = N/2-1, num = int(N/2));

    for i in range(0, len(window_range_1)):
        A[0][int(window_range_1[i])-1] = np.conj(A[0][int(window_range_2[i])-1])

    A[0][int(1+N/2)] = 0;
 
    B = scipy.fft.ifft(A);
    B = scipy.fft.fftshift(np.real(B));
    B = np.real(B);
    A = np.real(scipy.fft.fftshift(A));
  
    B = B / np.sum(B)
    coeff = np.reshape(B, (int(L), int(M)))
    coeff=np.transpose(coeff)                         
    return coeff

def padding(signal_in,pad_number):
    """
    

    Parameters
    ----------
    signal_in : TYPE
        DESCRIPTION.
    pad_number : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    pad_quantity =pad_number -  np.mod(len(signal_in),pad_number)
    return np.pad(signal_in,(0,pad_quantity))



def npr_analysis(coeff, x):
    """
    %NPR_ANALYSIS Near perfect reconstruction analysis filter bank.
    %  Y = NPR_ANALYSIS(COEFF,X) separates the input signal X into channels. 
    %  Each channel is a row in Y. COEFF is a two dimensional array containing
    %  the filter coefficients. The number of rows in Y will be twice the 
    %  number of rows in COEFF. 
    %
    %  See also npr_synthesis, npr_coeff.
    %
    % (c) 2007 - 2022 Wessel Lubberhuizen
    """
    x_temp = x.copy()
    coeff_temp = coeff.copy()

    
    # number of channels
    N = np.size(coeff_temp, 0)
    
    # number of slices
    if len(x_temp) > 1:
        x_temp = np.reshape(x_temp, (1, len(x_temp)), order = 'F')
    M = np.ceil(len(x_temp[0])/N)
    M = int(M)
    
    # number of samples
    L = M*N
    
    # create polyphase input signals
    x1 = np.reshape(x_temp, (int(N), int(M)), order = 'F')
    #x_temp = x.copy()
    
    # apply frequency shift
    phi = np.mod(np.linspace(start = 0, stop = L-1, num = L),2*N)*(np.pi/N)
    x2 = np.multiply(x_temp, np.exp(phi*0 + 1j * phi))
    x2 = np.reshape(x2, (int(N), int(M)), order = 'F')
    
    # apply channel filters
    coeff_temp = np.fliplr(coeff_temp)
    for i in np.arange(N):
        x1[i] = scipy.signal.lfilter(coeff_temp[i], 1, x1[i])
        x2[i] = scipy.signal.lfilter(coeff_temp[i], 1, x2[i])
    
    # apply dft
    y1 = scipy.fft.ifft(x1, n=None, axis=0)*N;
    y2 = scipy.fft.ifft(x2, n=None, axis=0)*N;
    
    # assemble even and off channels
    y = np.array((np.reshape(y1, (int(N), int(M)), order = 'F'), np.reshape(y2, (int(N), int(M)), order = 'F')))
    y = np.reshape(y, (2*N, M),order='F')
    
    return y




def npr_synthesis(coeff, y):
    """
    %NPR_SYNTHESIS Near perfect reconstruction synthesis filter bank.
    %  X = NPR_SYNTHESIS(C,Y) synthesizes a transient signal X from a number of
    %  channel transients stored in Y. Each channel is represented by a row.
    %  C is a two dimensional array, containing the filter coefficients.
    %  The number of rows in Y must be twice the number of rows in C. 
    %
    %  See also npr_analysis, npr_coeff.
    %
    % (c) 2007 - 2022 Wessel Lubberhuizen
    """
    y_temp = y.copy()
    coeff_temp = coeff.copy()
    
    # number of channels
    N=np.size(coeff,0);

    # number of slices
    M=np.size(y_temp,1);

    # number of samples
    L=M*N;
    
    # split into even and odd channels
    
    y_temp = np.reshape(y_temp, (2, N, M),order='F')
    y1 = np.squeeze(y_temp[0]);
    y2 = np.squeeze(y_temp[1]);
    
    # apply dft
    x1 = scipy.fft.fft(y1, n=None, axis=0)*N;
    x2 = scipy.fft.fft(y2, n=None, axis=0)*N;
    
    #apply channel filters
    for i in np.arange(N):
        x1[i] = scipy.signal.lfilter(coeff_temp[i], 1, x1[i])
        x2[i] = scipy.signal.lfilter(coeff_temp[i], 1, x2[i])
        
    # reshape time series
    x1 = np.reshape(x1, L,order='F')
    x2 = np.reshape(x2, L,order='F')
    
    # apply frequency shift
    phi = np.mod(np.linspace(start = 0, stop = L-1, num = L),2*N)*(np.pi/N)
    x2 = np.multiply(x2, np.exp(phi*0 - 1j * phi))
    
    # combine filter results
    
    x = x1 - x2
    x = np.reshape(x, (1, len(x)))
    
    return x



def demodulator_data_aided(signal_in,signal_ref,samples_per_symbol,frequency,sampling_freq,h_rccs,range_correlation = np.arange(1000,10000)):
    """
    Data aided demodulator

    Parameters
    ----------
    signal_in : TYPE
        DESCRIPTION.
    signal_ref : TYPE
        DESCRIPTION.
    samples_per_symbol : TYPE
        DESCRIPTION.
    frequency : TYPE
        DESCRIPTION.
    sampling_freq : TYPE
        DESCRIPTION.
    h_rccs : TYPE
        DESCRIPTION.
    range_correlation : TYPE, optional
        DESCRIPTION. The default is np.arange(1000,10000).

    Returns
    -------
    TYPE
        DESCRIPTION.
    Delay : TYPE
        DESCRIPTION.

    """
    
    signal_in_copy = signal_in.copy() # Create true copy 
    #Calculate delay between signals
    C_I,Gain, Delay, Signal_Interferences =CoverI(signal_ref[range_correlation],signal_in[range_correlation])
    signal_temp = signal.convolve(frequency_offset(signal_in, -frequency, sampling_freq), np.flipud(h_rccs),mode='same')
    signal_in = signal_in_copy.copy() # Restore signal
    return signal_temp[Delay::samples_per_symbol] , Delay

""" non data aided demodulator - for QPSK only """
def demodulator_non_data_aided(signal_in,samples_per_symbol,frequency,sampling_freq,h_rccs,range_delay = np.arange(0,10000)):
    """
    Non data aided demodulator for QPSK and 8PSK only
    signal is demodulated with various delay up to find the best SNR via M2M4 method
    
    Parameters
    ----------
    signal_in : numpy complex array
        input signal to be demodulated
    samples_per_symbol : integer
        number of samples per symbol
    frequency : float
        Centre frequency of the input signal
    sampling_freq : float
        sampling frequency
    h_rccs : TYPE
        reception filter
    range_delay : numpy array of int
        Delay to be checked. The default is np.arange(0,1000).

    Returns
    -------
    Numpy complex array
        Demodulated symbols
    best_delay : float
        Best delay

    """
    
    signal_in_copy = signal_in.copy() # Create true copy 
    if type(h_rccs) != float or type(h_rccs) != int:  
        signal_temp = signal.convolve(frequency_offset(signal_in, -frequency, sampling_freq), np.flipud(h_rccs),mode='same')
    best_SNR = -9999
    best_delay=0
    for i in range_delay:
        SNR_temp = M2M4(signal_temp[i::samples_per_symbol],1)
        
        if SNR_temp > best_SNR:
            best_SNR=SNR_temp
            best_delay=i
    signal_in = signal_in_copy.copy() # Restore signal        
    return signal_temp[best_delay::samples_per_symbol] , best_delay


def PAPR(signal_in):
    """
    Peak to average Power Ratio

    Parameters
    ----------
    signal_in : numpy array
        signal input

    Returns
    -------
    PAPR : float
        Peak to Average Power Ratio

    """
    Power_Mean_signal_in = np.mean(np.abs(signal_in)**2)
    Power_Peak_signal_in = np.max(np.abs(signal_in)**2)
    
    PAPR = 10*np.log10(Power_Peak_signal_in/Power_Mean_signal_in)
    return PAPR

def EVM(SNR,PAPR):
    EVM = 100 * 10**(-(SNR+PAPR)/20)
    return EVM

def ZOH(signal_in,upsampling_factor):
    """
    Zero Order Hold

    Parameters
    ----------
    signal_in : numpy array
        Input Signal
    upsampling_factor : float
        upsampling factor

    Returns
    -------
    signal_out : numpy array
        Zero order hold signal

    """
    signal_tmp = mp.repmat(signal_in,upsampling_factor,1)
    signal_out=np.reshape(signal_tmp,len(signal_in)*upsampling_factor,'F')
    return signal_out
