import numpy as np
from gnuradio import gr

class ADC_delta_sigma_sinus(gr.sync_block):  
    """ADC Delta Sigma Block"""

    def __init__(self, Vpp=2, enob=3, upsampling=64):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='ADC delta sigma sinus',   # will show up in GRC
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        self.Vpp = Vpp
        self.enob = enob
        self.upsampling = upsampling

    def ADC_Delta_Sigma(self, signal_in):
        signal_feedback = np.zeros(len(signal_in))
        signal_integrator = np.zeros(len(signal_in))
        signal_out = np.zeros(len(signal_in))
        
        # Loop
        for i in np.arange(0, len(signal_in)):
            # Delta
            if i == 0:
                signal_feedback[i] = signal_in[i]
            else:
                signal_feedback[i] = signal_in[i] - signal_out[i-1]
            
            # Intégrateur
            if i == 0:
                signal_integrator[i] = signal_feedback[i]
            else:
                signal_integrator[i] = signal_feedback[i] + signal_integrator[i-1]

            signal_integrator[i] = self.Vpp / (2**self.enob) * np.round(signal_integrator[i] / (self.Vpp / (2**self.enob)))
        
            # Comparateur
            if signal_integrator[i] >= 0:
                signal_out[i] = self.Vpp / 2
            else:
                signal_out[i] = -self.Vpp / 2
                
        return signal_out

    def work(self, input_items, output_items):
        signal_in = input_items[0]
        signal_out = self.ADC_Delta_Sigma(signal_in)
        output_items[0][:] = signal_out
        return len(output_items[0])