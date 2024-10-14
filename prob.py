import numpy as np
from scipy.signal import lfilter, ellip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def fix_aberrration(image_aberre):
    zeros = [0, -0.99, -0.99, 0.8]
    p1 = 0.9 * np.exp(1j * np.pi / 2)
    p2 = 0.9 * np.exp(-1j * np.pi / 2)
    p3 = 0.95 * np.exp(1j * np.pi / 8)
    p4 = 0.95 * np.exp(-1j * np.pi / 8)
    poles = [p1, p2, p3, p4]

    coef_numerateurs = np.poly(zeros)
    coef_denominateurs = np.poly(poles)

    clean_image = lfilter(coef_numerateurs, coef_denominateurs, image_aberre)
    return clean_image

def show_and_compare_images(image_deforme, clean_image, deformation_type):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_deforme, cmap='gray')
    plt.title(f"Image avec {deformation_type}")
    plt.axis('off')

    # Image après application du filtre inverse
    plt.subplot(1, 2, 2)
    plt.imshow(clean_image, cmap='gray')
    plt.title(f"Image sans {deformation_type}")
    plt.axis('off')

    plt.show()


def rotate_image_90_right(image):
    # Dimensions de l'image originale
    height, width = image.shape

    # Créer une nouvelle image pour la rotation (width et height sont inversés)
    rotated_image = np.zeros((width, height), dtype=image.dtype)

    # Matrice de rotation pour 90 degrés vers la droite
    rotation_matrix = np.array([[0, 1], [-1, 0]])

    # Appliquer la transformation de rotation
    for x in range(height):
        for y in range(width):
            # Nouvelle position (x', y') selon la matrice de rotation
            new_x, new_y = rotation_matrix @ np.array([x, y])
            # Ajuster les coordonnées dans le système de coordonnées de l'image tournée
            # new_x devient une colonne (donc y dans le nouvel espace)
            # new_y est inversé pour être compatible avec le système d'origine de l'image
            rotated_image[y, height - 1 - x] = image[x, y]

    return rotated_image

def filtre_methode_bilinieaire(image):
    fc = 500
    fe = 1600
    wd = 2 * np.pi * fc / fe
    wa = 2 * 1600 * np.tan(wd / 2) # Frequence de coupure normalise dans prob jpense
    x = 2 * fe / wa
    num1 = x ** 2 + np.sqrt(2) * x + 1
    coef_numerateurs = [1/num1, 2/num1, 1/num1]
    denom2 = (2 - 2 * (x**2)) / num1
    denom3 = (x**2 - np.sqrt(2)*x + 1) / num1
    coef_denominateurs = [1, denom2/num1, denom3/num1]
    filtre_image = lfilter(coef_numerateurs, coef_denominateurs, image)
    return filtre_image

def filtre_elliptic(image):
    Wn = 0.625
    N=4
    pass_ripple = 0.2
    stop_atten = 60  # Atténuation minimum en dehors de la bande passante en dB
    b, a = ellip(N=N, rp=pass_ripple, rs=stop_atten, Wn=Wn, btype='low', output='ba')
    clean_image = lfilter(b, a, image)
    return clean_image

def main():
    image_aberre = np.load("goldhill_aberrations.npy")
    clean_image = fix_aberrration(image_aberre)
    show_and_compare_images(image_aberre, clean_image, 'aberrations')

    image_bruite = np.load("goldhill_bruit.npy")
    image_filtrer_bilineaire = filtre_methode_bilinieaire(image_bruite)
    show_and_compare_images(image_bruite, image_filtrer_bilineaire, 'bruit (bilineaire)')

    image_filtrer_elliptic = filtre_elliptic(image_bruite)
    show_and_compare_images(image_bruite, image_filtrer_elliptic, 'bruit (elliptic)')
    plt.gray()
    image_a_tourner_couleur = mpimg.imread("goldhill_rotate.png")
    image_a_tourner = np.mean(image_a_tourner_couleur, -1)
    rotated_image = rotate_image_90_right(image_a_tourner)
    show_and_compare_images(image_a_tourner, rotated_image, 'mauvaise rotation')
if __name__ == "__main__":
    main()