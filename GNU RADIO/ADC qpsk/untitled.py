#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: IvanB
# GNU Radio version: 3.10.10.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import sip
import untitled_epy_block_0 as epy_block_0  # embedded python block
import untitled_epy_block_0_0 as epy_block_0_0  # embedded python block
import untitled_epy_block_1 as epy_block_1  # embedded python block
import untitled_epy_block_1_0 as epy_block_1_0  # embedded python block
import untitled_epy_block_3 as epy_block_3  # embedded python block
import untitled_epy_block_4 as epy_block_4  # embedded python block



class untitled(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "untitled")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.time = time = 0.5
        self.fs = fs = 8000
        self.upsampling = upsampling = 64
        self.syms = syms = 10
        self.samples_per_symbol = samples_per_symbol = 20
        self.qpsk = qpsk = [-1-1j,+1-1j,1+1j,-1+1j]
        self.noise = noise = 0.1
        self.n_samples = n_samples = fs*time
        self.fp = fp = fs/10
        self.enob = enob = 3
        self.cutoff_freq = cutoff_freq = 600
        self.beta = beta = 0.2
        self.Vpp = Vpp = 2

        ##################################################
        # Blocks
        ##################################################

        self._noise_range = qtgui.Range(0, 0.5, 0.001, 0.1, 200)
        self._noise_win = qtgui.RangeWidget(self._noise_range, self.set_noise, "'noise'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._noise_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            fs, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
            1024, #size
            "Constellation after adc", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0.enable_grid(False)
        self.qtgui_const_sink_x_0_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_0_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "Constellation before adc", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(False)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_win)
        self.filter_fft_low_pass_filter_0 = filter.fft_filter_ccc(upsampling, firdes.low_pass(1, fs, cutoff_freq, 1, window.WIN_HAMMING, 6.76), 1)
        self.epy_block_4 = epy_block_4.QPSKDemodulator(syms=syms, beta=beta, P=samples_per_symbol, nb_symbols=n_samples)
        self.epy_block_3 = epy_block_3.blk()
        self.epy_block_1_0 = epy_block_1_0.blk(fs=fs, frequency=-fp)
        self.epy_block_1 = epy_block_1.blk(fs=fs, frequency=fp)
        self.epy_block_0_0 = epy_block_0_0.blk(Vpp=Vpp, enob=enob, upsampling=upsampling)
        self.epy_block_0 = epy_block_0.blk(syms=syms, beta=beta, P=samples_per_symbol)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc(qpsk, 1)
        self.blocks_repeat_0 = blocks.repeat(gr.sizeof_float*1, upsampling)
        self.blocks_packed_to_unpacked_xx_0 = blocks.packed_to_unpacked_bb(2, gr.GR_MSB_FIRST)
        self.blocks_complex_to_real_0 = blocks.complex_to_real(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 255, 1000))), True)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, noise, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_packed_to_unpacked_xx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.epy_block_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.blocks_complex_to_real_0, 0), (self.blocks_repeat_0, 0))
        self.connect((self.blocks_packed_to_unpacked_xx_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_repeat_0, 0), (self.epy_block_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.epy_block_0, 0), (self.epy_block_1, 0))
        self.connect((self.epy_block_0_0, 0), (self.epy_block_3, 0))
        self.connect((self.epy_block_1, 0), (self.blocks_complex_to_real_0, 0))
        self.connect((self.epy_block_1, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.epy_block_1_0, 0), (self.epy_block_4, 0))
        self.connect((self.epy_block_3, 0), (self.filter_fft_low_pass_filter_0, 0))
        self.connect((self.epy_block_4, 0), (self.qtgui_const_sink_x_0_0, 0))
        self.connect((self.filter_fft_low_pass_filter_0, 0), (self.epy_block_1_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "untitled")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_time(self):
        return self.time

    def set_time(self, time):
        self.time = time
        self.set_n_samples(self.fs*self.time)

    def get_fs(self):
        return self.fs

    def set_fs(self, fs):
        self.fs = fs
        self.set_fp(self.fs/10)
        self.set_n_samples(self.fs*self.time)
        self.epy_block_1.fs = self.fs
        self.epy_block_1_0.fs = self.fs
        self.filter_fft_low_pass_filter_0.set_taps(firdes.low_pass(1, self.fs, self.cutoff_freq, 1, window.WIN_HAMMING, 6.76))
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.fs)

    def get_upsampling(self):
        return self.upsampling

    def set_upsampling(self, upsampling):
        self.upsampling = upsampling
        self.blocks_repeat_0.set_interpolation(self.upsampling)
        self.epy_block_0_0.upsampling = self.upsampling

    def get_syms(self):
        return self.syms

    def set_syms(self, syms):
        self.syms = syms
        self.epy_block_0.syms = self.syms
        self.epy_block_4.syms = self.syms

    def get_samples_per_symbol(self):
        return self.samples_per_symbol

    def set_samples_per_symbol(self, samples_per_symbol):
        self.samples_per_symbol = samples_per_symbol
        self.epy_block_0.P = self.samples_per_symbol
        self.epy_block_4.P = self.samples_per_symbol

    def get_qpsk(self):
        return self.qpsk

    def set_qpsk(self, qpsk):
        self.qpsk = qpsk
        self.digital_chunks_to_symbols_xx_0.set_symbol_table(self.qpsk)

    def get_noise(self):
        return self.noise

    def set_noise(self, noise):
        self.noise = noise
        self.analog_noise_source_x_0.set_amplitude(self.noise)

    def get_n_samples(self):
        return self.n_samples

    def set_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.epy_block_4.nb_symbols = self.n_samples

    def get_fp(self):
        return self.fp

    def set_fp(self, fp):
        self.fp = fp
        self.epy_block_1.frequency = self.fp
        self.epy_block_1_0.frequency = -self.fp

    def get_enob(self):
        return self.enob

    def set_enob(self, enob):
        self.enob = enob
        self.epy_block_0_0.enob = self.enob

    def get_cutoff_freq(self):
        return self.cutoff_freq

    def set_cutoff_freq(self, cutoff_freq):
        self.cutoff_freq = cutoff_freq
        self.filter_fft_low_pass_filter_0.set_taps(firdes.low_pass(1, self.fs, self.cutoff_freq, 1, window.WIN_HAMMING, 6.76))

    def get_beta(self):
        return self.beta

    def set_beta(self, beta):
        self.beta = beta
        self.epy_block_0.beta = self.beta
        self.epy_block_4.beta = self.beta

    def get_Vpp(self):
        return self.Vpp

    def set_Vpp(self, Vpp):
        self.Vpp = Vpp
        self.epy_block_0_0.Vpp = self.Vpp




def main(top_block_cls=untitled, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
