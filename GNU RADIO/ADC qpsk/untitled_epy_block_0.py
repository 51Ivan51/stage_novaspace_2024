import numpy as np
from gnuradio import gr
from scipy import signal


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Convolve Block"""

    def __init__(self, syms=10, beta=0.2, P=20):  # only default arguments here
        """Arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Convolve',  # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.syms = syms
        self.beta = beta
        self.P = P

    def root_raised_cos(self):
        k = np.arange(-self.syms * self.P + 1e-8, (self.syms * self.P + 1e-8) + 1)
        s = (np.sin(np.pi * (1 - self.beta) * k / self.P) +
             4 * self.beta * k / self.P * np.cos(np.pi * k / self.P * (1 + self.beta))) / \
            (np.pi * k / self.P * (1 - 16 * (self.beta**2) * (k / self.P)**2))
        s = s / np.sqrt(np.sum(np.abs(s)**2))
        return s

    def work(self, input_items, output_items):
        h_rrc = self.root_raised_cos()
        signal_qpsk = signal.convolve(input_items[0], h_rrc, mode='same')
        output_items[0][:] = signal_qpsk
        return len(output_items[0])
