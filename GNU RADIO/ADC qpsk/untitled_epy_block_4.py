import numpy as np
from gnuradio import gr
from scipy import signal

class QPSKDemodulator(gr.sync_block):
    """QPSK Demodulation Block"""

    def __init__(self, syms=10, beta=0.2, P=20, nb_symbols=4000):
        """Constructor for QPSK Demodulation Block"""
        gr.sync_block.__init__(
            self,
            name='QPSK Demodulator',  # Block name displayed in GRC
            in_sig=[np.complex64],     # Input signal type (complex64)
            out_sig=[np.complex64]     # Output signal type (complex64)
        )
        self.syms = syms      # Number of symbols
        self.beta = beta      # Roll-off factor of RRC filter
        self.P = P            # Oversampling factor
        self.nb_symbols = nb_symbols  # Number of QPSK symbols

    def root_raised_cos(self):
        k = np.arange(-self.syms * self.P, self.syms * self.P + 1e-8)
        s = (np.sin(np.pi * (1 - self.beta) * k / self.P) +
             4 * self.beta * k / self.P * np.cos(np.pi * k / self.P * (1 + self.beta))) / \
            (np.pi * k / self.P * (1 - 16 * (self.beta**2) * (k / self.P)**2))
        s = s / np.sqrt(np.sum(np.abs(s)**2))
        return s

    def qpsk(self):
        h_rrc = self.root_raised_cos()
        symbols = np.random.randint(4, size=self.nb_symbols)
        signal_raw = np.repeat(np.exp(1j * np.pi * (symbols / 2 + 1/4)), self.P)
        delay = (len(h_rrc) - 1) // 2
        signal_qpsk = signal.convolve(signal_raw, h_rrc, mode='same')
        return signal_qpsk, delay

    def work(self, input_items, output_items):
        h_rrc = self.root_raised_cos()
        h_rrc_inv = np.flipud(h_rrc)
        demod_convolv = signal.convolve(input_items[0], h_rrc_inv, mode='same')
        symbols_out_demod = demod_convolv[0::self.P]

        output_items[0][:] = symbols_out_demod
        return len(output_items[0])
