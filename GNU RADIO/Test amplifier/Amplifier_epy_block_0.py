import numpy as np
from gnuradio import gr

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self, hpa=1):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Amplifiers',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.hpa = np.array(hpa)

    def amplifier(self, signal_in):
        ibo_range = self.hpa[:, 0]
        obo_range = self.hpa[:, 1]
        phase_range = self.hpa[:, 2]
        signal_in_am = 20 * np.log10(np.abs(signal_in))
        signal_in_pm = np.angle(signal_in)
        signal_out_am = np.interp(signal_in_am, ibo_range, obo_range)
        signal_out_pm = np.interp(signal_in_am, ibo_range, phase_range) * np.pi / 180
        signal_out = 10**(signal_out_am / 20) * np.exp(1j * (signal_in_pm + signal_out_pm))
        
        return signal_out
    
    def work(self, input_items, output_items):
        signal_in = input_items[0]
        signal_out = self.amplifier(signal_in)

        output_items[0][:] = signal_out

        return len(output_items[0])
