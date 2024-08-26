import numpy as np
from scipy.fft import fft, fftfreq
from bokeh.plotting import figure, show, gridplot

# Définition de la fonction `amplifier`
def amplifier(signal_in, hpa):
    """
    Parameters
    ----------
    signal_in : numpy array
        Input signal
    hpa : numpy array
        HPA AMAM/AMPM model file content.
        Column 1 = IBO, Column 2 = OBO, Column 3 = Phase shift

    Returns
    -------
    signal_out : numpy array
        Complex signal after amplification
    """
    
    ibo_range = hpa[:,0]
    obo_range = hpa[:,1]
    phase_range = hpa[:,2]
    
    signal_in_am = 20 * np.log10(np.abs(signal_in))
    signal_in_pm = np.angle(signal_in)
    signal_out_am = np.interp(signal_in_am, ibo_range, obo_range)
    signal_out_pm = np.interp(signal_in_am, ibo_range, phase_range) * np.pi / 180

    signal_out = 10**(signal_out_am / 20) * np.exp(1j * (signal_in_pm + signal_out_pm))
    
    return signal_out

# Modèle HPA simulé
hpa_model = np.array([
    [0, -1, 0],
    [1, 0, 5],
    [2, 1, 10],
    [3, 2, 15],
    [4, 3, 20],
    [5, 4, 25]
])

# Génération d'un signal avec deux sinusoïdes complexes
fs = 500  # fréquence d'échantillonnage
t = np.arange(0, 1, 1/fs)  # vecteur temps
freq1 = 5  # fréquence du premier signal
freq2 = 50  # fréquence du deuxième signal
signal_in = np.exp(1j * 2 * np.pi * freq1 * t) + np.exp(1j * 2 * np.pi * freq2 * t) + 0.1 * (np.random.randn(t.size) + 1j * np.random.randn(t.size))

# Application de l'amplification
signal_out = amplifier(signal_in, hpa_model)

# Calcul de la FFT
N = len(signal_in)
f = fftfreq(N, 1/fs)

fft_in = fft(signal_in)
fft_out = fft(signal_out)

# Visualisation du signal d'entrée et du signal amplifié
p1 = figure(title="Signal d'entrée", x_axis_label='Temps (s)', y_axis_label='Amplitude')
p1.line(t, np.abs(signal_in), legend_label="Signal d'entrée", line_width=2, color='blue', alpha=0.5)

p2 = figure(title="Signal amplifié", x_axis_label='Temps (s)', y_axis_label='Amplitude')
p2.line(t, np.abs(signal_out), legend_label="Signal amplifié", line_width=2, color='red')

# Visualisation de la FFT
p3 = figure(title="FFT du signal d'entrée", x_axis_label='Fréquence (Hz)', y_axis_label='Amplitude')
p3.line(f[:N//2], np.abs(fft_in)[:N//2], legend_label="FFT Signal d'entrée", line_width=2, color='blue')

p4 = figure(title="FFT du signal amplifié", x_axis_label='Fréquence (Hz)', y_axis_label='Amplitude')
p4.line(f[:N//2], np.abs(fft_out)[:N//2], legend_label="FFT Signal amplifié", line_width=2, color='red')

# Organisation des graphiques en grille
grid = gridplot([[p1, p2], [p3, p4]])

# Affichage des figures
show(grid)
