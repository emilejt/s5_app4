
import zplane
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq


#1.a)
zeros = [0.8j, -0.8j]
p1 = 0.95 * np.exp(1j * np.pi / 8)
p2 = 0.95 * np.exp(-1j * np.pi / 8)
poles = [p1,p2]

a = np.poly(poles) #Coef denominateur

b = np.poly(zeros) #coef num

#zplane.zplane(b, a)

#1.c)
w, h = signal.freqz(b, a, worN = 8000)

h_db = 20 * np.log10(np.abs(h))

plt.figure()
plt.plot(w / np.pi, h_db) #Diviser par pi pour avoir la freq normalis/e entre 0 et 1
plt.title('Reponse en frequence du filtre')
plt.xlabel(r'Frequence normalisee ($\omega / \pi$)')
plt.ylabel('Module (dB)')
plt.grid()

plt.savefig('reponse-en_frequence.png')
plt.show()

#1.D)

#Parametres
N=512
impulse = np.zeros(N)
impulse[N // 2] = 1

#Filter impulsion
h_n = signal.lfilter(b, a, impulse)

#FFT du h(n)
H_n = fft(h_n)
frequencies = fftfreq(N, d=1)[:N // 2]  # Fréquences positives uniquement

w, H_w = signal.freqz(b, a, worN=N)

plt.figure()

plt.subplot(1,2,1)
plt.plot(w / np.pi, 20 * np.log10(abs(H_w)), label="Reponse en frequence du filtre")


# Module de la transformée de Fourier de h[n] (normalisé)
plt.plot(frequencies / (frequencies.max()), 20 * np.log10(np.abs(H_n[:N // 2])), label='Transformée de Fourier de la réponse impulsionnelle |h[n]|')
# Configuration du graphique
plt.title('Comparaison de la réponse en fréquence et de la réponse impulsionnelle')
plt.xlabel(r'Fréquence normalisée ($\omega / \pi$)')
plt.ylabel('Module (dB)')
plt.legend()
plt.grid()
plt.show()