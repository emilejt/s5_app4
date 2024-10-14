import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt

zeros = [0, -0.99, -0.99, 0.8]
p1 = 0.9 * np.exp(1j*np.pi/2)
p2 = 0.9 * np.exp(-1j*np.pi/2)
p3 = 0.95 * np.exp(1j*np.pi/8)
p4 = 0.95 * np.exp(-1j*np.pi/8)
poles = [p1, p2, p3, p4]

coef_numerateurs = np.poly(zeros)
coef_denominateurs = np.poly(poles)
image_aberre = np.load("goldhill_aberrations.npy")
height, width = image_aberre.shape

clean_image = lfilter(coef_numerateurs, coef_denominateurs, image_aberre)



plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(image_aberre, cmap='gray')
plt.title("Image avec aberrations")
plt.axis('off')

# Image après application du filtre inverse
plt.subplot(1, 2, 2)
plt.imshow(clean_image, cmap='gray')
plt.title("Image après correction")
plt.axis('off')

plt.show()