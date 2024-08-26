import numpy as np
from gnuradio import gr
from scipy import signal

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple lowpass filter"""

    def __init__(self, fs=8000, cutoff_freq=600, upsampling = 64):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='low pass filter + decimation',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.fs = fs
        self.cutoff_freq = cutoff_freq
        self.upsampling = upsampling

    def lowpass_filter(self, signal_in):
        nyquist = self.fs / 2
        normal_cutoff = self.cutoff_freq / nyquist
        nb_coeff = 101
        coeff_FIR = signal.firwin(nb_coeff, normal_cutoff, window='hamming')
        filtered_signal = signal.convolve(signal_in, coeff_FIR, mode='same')
        return filtered_signal

    def work(self, input_items, output_items):

        signal_filtered = self.lowpass_filter(input_items[0])

        output_items[0][:] = signal.decimate(signal_filtered,upsampling,ftype ='fir')

        return len(output_items[0])
