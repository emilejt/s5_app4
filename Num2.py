import zplane
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

z1 = np.exp(1j * np.pi / 16)
z2 = np.exp(-1j * np.pi / 16)

zeros = [z1,z2]
poles = [0.95, 0.95]

a = np.poly(poles) #Coef denominateur

b = np.poly(zeros) #coef num

w, h = signal.freqz(b, a, worN = 8000)

h_db = 20 * np.log10(np.abs(h))

plt.figure()
plt.plot(w / np.pi, h_db) #Diviser par pi pour avoir la freq normalis/e entre 0 et 1
plt.title('Reponse en frequence du filtre')
plt.xlabel(r'Frequence normalisee ($\omega / \pi$)')
plt.ylabel('Module (dB)')
plt.grid()
plt.show()

N = 2048
n = np.arange(N)
x_n = np.sin(n*np.pi / 16) + np.sin(n*np.pi / 32)
y_n = signal.lfilter(b, a, x_n)

plt.subplot(2,1,1)
plt.plot(n, x_n)
plt.title('Signal d entree x[n]')
plt.xlabel('n')
plt.ylabel('amplitude')
plt.grid()

plt.subplot(2,1,2)
plt.plot(n, y_n)
plt.title('x_n de sortie y[n]')
plt.xlabel('n')
plt.ylabel('amplitude')
plt.grid()

plt.tight_layout()
plt.show()