import numpy as np
from gnuradio import gr

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self, fs=8000, frequency=800):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Frequency offset',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.fs = fs
        self.frequency = frequency

    def frequency_offset(self, signal_in):
        time = np.arange(0, signal_in.size) / self.fs
        signal_LO = np.exp(1j * 2 * np.pi * self.frequency * time)
        return signal_in * signal_LO

    def work(self, input_items, output_items):
        output_items[0][:] = self.frequency_offset(input_items[0])
        return len(output_items[0])
