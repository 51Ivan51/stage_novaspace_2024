# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:55:53 2023

@author: nicol
"""

import numpy as np
import usual_NJ
import usual
import matplotlib.pyplot as plt
from scipy import signal
import Filter_model

# Simulation inputs 
modulation = 'QPSK'
samples_per_symbol = 120;
frequency = 0.35e9;
sampling_freq  = 25.56e9;
Ns = 1000000;

# IPF inputs
cutoff_freq_IPF = (6623.42e6, 6708.45e6)  # Cutoff frequency of the filter
order_IPF = 4  # Filter order
passband_ripple_IPF = 0.1  # Passband ripple in dB
Insertion_losses_IPF = 0.5 # Filter insertion losses

b_IPF, a_IPF = Filter_model.design_chebyshev_bandpass_filter(order_IPF, passband_ripple_IPF, cutoff_freq_IPF, sampling_freq)


signal_test=np.zeros(1000)
signal_test[500]=1

#create signal input
input_signal, delay, h_rccs = usual.qpsk(Ns, 0.2, samples_per_symbol)
input_signal = usual.normalize(input_signal)
input_signal = usual.frequency_offset(input_signal, frequency, sampling_freq)
input_signal_ref = input_signal
#input_signal = usual.add_awgn_to_complex_signal(input_signal, 0.1)

#filter via lfilter

output_signal_temp = usual.ADC(input_signal,sampling_freq,1.42e9,10,10)
output_signal_temp = usual.padding(output_signal_temp,2**14)
#output_signal_temp = input_signal


# print(usual.power_mean(output_signal_temp))
#Define polyphase coefficient
coeff = usual.npr_coeff(2**14, 64)
coeff = coeff / np.linalg.norm(coeff)

# Go through polyphase filter bank
signal_DEMUX = usual.npr_analysis(coeff, output_signal_temp)
signal_MUX = usual.npr_synthesis(coeff,signal_DEMUX)
signal_MUX=np.squeeze(signal_MUX)
signal_MUX = usual.normalize(signal_MUX)

# up sample between ADC and DAC
window=[1]

output_signal_temp1 = signal.upfirdn(window, signal_MUX,up=6,down=1)

# Go through DAC
output_signal_temp1=usual.DAC(output_signal_temp1,1,3,10,10)

# Remove low level first samples
a=np.where(np.abs(output_signal_temp1[1:-1])>0.05*np.max(np.abs(output_signal_temp1)))
b=min(a)

# recover symbols
symbols,delay = usual.demodulator_non_data_aided(output_signal_temp1[b[1]:] , samples_per_symbol, frequency, sampling_freq, h_rccs)

plt.scatter(np.real(symbols),np.imag(symbols))

# output_signal_temp = usual.DAC(output_signal_temp,1.42e9,25.56e9,10,10)
# # # print(len(output_signal_temp))

# symbols,delay = usual.demodulator_non_data_aided(output_signal_temp , samples_per_symbol, frequency, sampling_freq, h_rccs)
# plt.scatter(np.real(symbols),np.imag(symbols))

# usual.pwelch(output_signal_temp,25.56e9)

