# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:10:01 2023

@author: nicol
"""

import usual
import numpy as np
from scipy import signal
import time
from numba import jit, float64, complex128
import matplotlib.pyplot as plt
from numpy import matlib as mp



# def ZOH(signal_in,upsampling_factor):
#     y = [0] * (upsampling_factor * len(signal_in))
#     for i in np.arange(upsampling_factor):
#         y[i::upsampling_factor] = signal_in
        
#     y=np.array(y)    
#     return y

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

"""
def ADC_sigma_delta(signal_in,upsampling,Vpp,enob,loopback_factor):
    signal_in_up = np.real(ZOH(signal_in,upsampling))
    #signal_in_up = ZOH(signal_in,upsampling)
    signal_out = np.zeros(len(signal_in_up))
    signal_feedback = np.zeros(len(signal_in_up))
    signal_integrator = np.zeros(len(signal_in_up))
    
    #Sigma Delta Loop
    for i in np.arange(1,len(signal_in_up)):
        #Feedback loop
        signal_feedback[i] = signal_in_up[i] + loopback_factor * signal_out[i-1]
        #integrator
        signal_integrator[i] = np.sum(signal_feedback[0:i])
        # quantizer
        signal_out[i]= np.clip(usual.quantizer(signal_integrator[i],Vpp/(2**enob)),-Vpp/2,Vpp/2)
        
        
    # FIR decimation and Hilbert transform 
    signal_out_final = signal.decimate(signal_out,upsampling,ftype ='fir')
    
    signal_out_final = signal.hilbert(signal_out_final)
    return signal_out_final,signal_out


def ADC_sigma_delta_2(signal_in,upsampling,Vpp,enob,loopback_factor):
    signal_in_up = np.real(ZOH(signal_in,upsampling))
    #signal_in_up = ZOH(signal_in,upsampling)
    signal_out = np.zeros(len(signal_in_up))
    signal_feedback = 0
    signal_integrator = 0
    
    #Sigma Delta Loop
    for i in np.arange(1,len(signal_in_up)):
        # Feedback loop
        signal_feedback = signal_in_up[i] + loopback_factor * signal_out[i-1]
        # Integrator
        signal_integrator += signal_feedback 
        # Quantizer
        signal_out[i]= np.clip(usual.quantizer(signal_integrator,Vpp/(2**enob)),-Vpp/2,Vpp/2)
        
        
    # FIR decimation and Hilbert transform 
    signal_out_final = signal.decimate(signal_out,upsampling,ftype ='fir')
    
    signal_out_final = signal.hilbert(signal_out_final)
    return signal_out_final,signal_out

"""

@jit(nopython=True)
def Sigma_Delta_Loop(signal_in_up,Vpp,enob,loopback_factor):
    """
    Sigma Delta loop
    Done via jit to accelerate

    Parameters
    ----------
    signal_in_up : TYPE
        input signal upsampled
    Vpp : float
        Voltage Peak to Peak
    enob : float
        Effective Number of Bits
    loopback_factor : float
        -1 by default. Shall be negative

    Returns
    -------
    signal_out : Numpy array
        Sigma Delta signal

    """
    # Init integrator and loopback
    signal_feedback = 0
    signal_integrator = 0
    
    signal_out = np.zeros(len(signal_in_up))
    
        #Sigma Delta Loop
    for i in np.arange(1,len(signal_in_up)):
        #Feedback loop
        signal_feedback = signal_in_up[i] + loopback_factor * signal_out[i-1]
        #integrator
        signal_integrator += signal_feedback
        # Quantizer
        signal_out[i] =Vpp/(2**enob) *  np.round(signal_integrator/(Vpp/(2**enob)))
        
        # Clipping
        if signal_out[i] > Vpp/2:
            signal_out[i] = Vpp/2
    
        if signal_out[i] < Vpp/2:
            signal_out[i] = -Vpp/2
            
    return signal_out

def ADC_sigma_delta(signal_in,upsampling,Vpp,enob,loopback_factor):
     # Zero Order Hold 
     signal_in_up=np.real(ZOH(signal_in,upsampling))
     
       
     # Call sigma delta loop
     signal_out =  Sigma_Delta_Loop(signal_in_up,Vpp,enob,loopback_factor)
     
     # FIR decimation and Hilbert transform 
     signal_out_final = signal.hilbert(signal.decimate(signal_out,upsampling,ftype ='fir'))
     
     return signal_out_final,signal_out

Vpp =1.8
enob = 3
upsampling = 10


loopback_factor = -1

signal_in,delay,h_rccs,symbols_in = usual.apsk16(1000,0.2,20)
signal_in=usual.frequency_offset(signal_in,0.3,1)

"""
start_time=time.time()
signal_out_final1,signal_out1 = ADC_sigma_delta_2(signal_in, upsampling, Vpp, enob, loopback_factor)
print("--- %s seconds ---" % (time.time() - start_time))
"""

# signal_in_up=np.real(ZOH(signal_in,upsampling))
# signal_out = np.zeros(len(signal_in_up))
# signal_feedback = np.zeros(len(signal_in_up))
# signal_integrator = np.zeros(len(signal_in_up))

# start_time=time.time()
# out2 =test_accelerator(signal_in_up,signal_out,signal_feedback,signal_integrator,Vpp,enob,loopback_factor)
# print("--- %s seconds ---" % (time.time() - start_time))

start_time=time.time()
signal_out_final2,signal_out2=ADC_sigma_delta(signal_in,upsampling,Vpp,enob,loopback_factor)
print("--- %s seconds ---" % (time.time() - start_time))


"""
symbols,delay = usual.demodulator_non_data_aided(usual.frequency_offset(signal_out_final1,-0.2,1),20,0,1,h_rccs)
plt.scatter(np.real(symbols),np.imag(symbols))
"""


symbols,delay = usual.demodulator_non_data_aided(signal_out_final2,20,0.3,1,h_rccs)
plt.scatter(np.real(symbols),np.imag(symbols))

# signal_in_up=np.real(ZOH(signal_in,upsampling))

# signal_out=signal_in_up.copy()

# signal_out=0*signal_out
# signal_temp = 0
# signal_temp_out =0
# signal_feedback=signal_out.copy()
# signal_integrator = signal_out.copy()

# #Sigma Delta Loop
# for i in np.arange(1,len(signal_in_up)):
#     #Feedback loop
#     signal_feedback[i] = signal_in_up[i] + loopback_factor * signal_out[i-1]
#     #integrator
#     signal_integrator[i] = np.sum(signal_feedback[0:i])
#     # ADC 
#     signal_out[i]== np.clip(usual.quantizer(signal_integrator[i],Vpp/(2**enob)),-Vpp/2,Vpp/2)
    

# # FIR decimation    
# signal_out_final = signal.decimate(signal_out,upsampling,ftype ='fir')