import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, cheby2, ellip, freqz, buttord, cheb1ord, cheb2ord, ellipord

# Spécifications du filtre
fs = 1600  # Fréquence d'échantillonnage en Hz
fc_pass = 500  # Fréquence de coupure de la bande passante en Hz
fc_stop = 750  # Fréquence de coupure de l'arrêt en Hz
pass_ripple = 0.2  # Tolérance du gain dans la bande passante en dB
stop_atten = 60  # Atténuation minimum en dehors de la bande passante en dB

# Normaliser les fréquences par rapport à la fréquence d'échantillonnage
wp = fc_pass / (0.5 * fs)  # Fréquence de coupure normalisée pour le passage
ws = fc_stop / (0.5 * fs)  # Fréquence de coupure normalisée pour l'arrêt

# Définir les types de filtres et leurs fonctions SciPy, avec leurs fonctions d'ordre respectives
filter_types = {
    "Butterworth": (butter, buttord),
    "Chebyshev I": (cheby1, cheb1ord),
    "Chebyshev II": (cheby2, cheb2ord),
    "Elliptic": (ellip, ellipord)
}

# Créer un graphique pour comparer les réponses en fréquence
plt.figure(figsize=(10, 8))

for filter_name, (filter_func, ord_func) in filter_types.items():
    # Calculer l'ordre minimal requis pour chaque type de filtre
    N, Wn = ord_func(wp, ws, gpass=pass_ripple, gstop=stop_atten)

    # Afficher l'ordre et la fréquence de coupure ajustée pour chaque filtre
    print(f"{filter_name} Filter - Order: {N}, Cutoff Frequency (normalized): {Wn}")

    # Calculer les coefficients du filtre
    if filter_name == "Butterworth":
        b, a = filter_func(N=N, Wn=Wn, btype='low', output='ba')
    elif filter_name == "Chebyshev I":
        b, a = filter_func(N=N, rp=pass_ripple, Wn=Wn, btype='low', output='ba')
    elif filter_name == "Chebyshev II":
        b, a = filter_func(N=N, rs=stop_atten, Wn=Wn, btype='low', output='ba')
    elif filter_name == "Elliptic":
        b, a = filter_func(N=N, rp=pass_ripple, rs=stop_atten, Wn=Wn, btype='low', output='ba')

    # Calculer la réponse en fréquence
    w, h = freqz(b, a, worN=8000)
    freq_hz = (fs * 0.5 / np.pi) * w

    # Tracer la réponse en dB
    plt.plot(freq_hz, 20 * np.log10(np.abs(h)), label=f"{filter_name} (ordre {N})")

# Configurer le graphique
plt.title("Comparaison des réponses en fréquence des différents filtres")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.ylim([-100, 5])
plt.legend()
plt.grid()
plt.show()
