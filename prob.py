import zplane
import numpy as np
from scipy.signal import lfilter, ellip, freqz
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
    zplane.zplane(coef_numerateurs, coef_denominateurs, title="poles et zeros de la fonction inverse (aberrations)")
    clean_image = lfilter(coef_numerateurs, coef_denominateurs, image_aberre)
    return clean_image

def show_and_compare_images(image_deforme, clean_image, deformation_type):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_deforme, cmap='gray')
    if deformation_type == 'rotation':
        plt.title("Image sans rotation")
    else:
        plt.title(f"Image avec {deformation_type}")
    plt.axis('off')

    # Image après application du filtre inverse
    plt.subplot(1, 2, 2)
    plt.imshow(clean_image, cmap='gray')
    if deformation_type == 'rotation':
        plt.title("Image avec rotation")
    else:
        plt.title(f"Image sans {deformation_type}")
    plt.axis('off')

    plt.show()

def rotate_image_90_right(image):

    # Obtenir les dimensions de l'image
    width, height = image.shape[:2]


    angle_rad = 270/180 * np.pi

    # Matrice de rotation avec l'angle de rotation de 270
    rotation_matrix = np.array([[int(np.cos(angle_rad)), int(-np.sin(angle_rad))], [int(np.sin(angle_rad)), int(np.cos(angle_rad))]])

    # Créer une nouvelle image avec les dimensions inversées (car 90° de rotation)
    rotated_image = np.zeros((height, width), dtype=image.dtype)

    # Appliquer la rotation pixel par pixel
    for col in range(width):
        for row in range(height):

            corrected_row = height - 1 - row
            # Calculer la nouvelle position du pixel après la rotation
            new_position = np.dot(rotation_matrix, [col, corrected_row])
            new_i, new_j = new_position[0], new_position[1]
            

            rotated_image[int(new_i), int(new_j)] = image[col, corrected_row]

    return rotated_image

def rotation_lineaire(image):

    width, height = image.shape[:2]
    rotated_image = np.zeros((height, width))


    for e1 in range(width):
        for e2 in range(height):
            u1 = e2
            u2 = -e1
            rotated_image[u1][u2] = image[e1][e2]

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

    zplane.zplane(coef_numerateurs, coef_denominateurs, title="poles et zeros du filtre bilineaire")
    plot_filter_frequency_response(coef_numerateurs, coef_denominateurs, 'Reponse en frequence filtre methode bilineaire')
    filtre_image = lfilter(coef_numerateurs, coef_denominateurs, image)
    return filtre_image

def filtre_elliptic(image):
    Wn = 0.625
    N=4
    pass_ripple = 0.2
    stop_atten = 60  # Atténuation minimum en dehors de la bande passante en dB
    b, a = ellip(N=N, rp=pass_ripple, rs=stop_atten, Wn=Wn, btype='low', output='ba')

    zplane.zplane(b, a, title="poles et zeros du filtre elliptic")
    plot_filter_frequency_response(b, a, 'Reponse en frequence filtre elliptic')
    clean_image = lfilter(b, a, image)
    return clean_image


def plot_filter_frequency_response(coef_numerateurs, coef_denominateurs, title="Réponse en fréquence"):
    """
    Affiche le module de la réponse en fréquence du filtre.

    Parameters:
    - coef_numerateurs: list, coefficients du numérateur du filtre.
    - coef_denominateurs: list, coefficients du dénominateur du filtre.
    - title: str, titre du graphique.
    """
    # Calculer la réponse en fréquence
    w, h = freqz(coef_numerateurs, coef_denominateurs, worN=8000)

    # Convertir la fréquence angulaire en Hz
    fe = 1600  # Fréquence d'échantillonnage en Hz
    freq = w * fe / (2 * np.pi)

    # Tracer le module de la réponse en fréquence en dB
    plt.figure(figsize=(10, 6))
    plt.plot(freq, 20 * np.log10(abs(h)), 'b')
    plt.title(title)
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim([-100, 5])  # Ajustez cette limite selon votre filtre pour une meilleure visualisation
    plt.show()

def compress(image, factor=0.5):
    # Calculer la matrice de covariance
    covariance = np.cov(image, rowvar=False)

    # Calculer les valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Trier les vecteurs propres par valeurs propres décroissantes
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Construire la matrice de passage avec les vecteurs propres triés
    transferMatrix = sorted_eigenvectors.T

    # Choisir les vecteurs propres les plus significatifs selon le facteur
    num_components = int(len(eigenvalues) * (1-factor))
    compression_eigenvectors = transferMatrix[:num_components]

    # Projeter l'image dans la nouvelle base
    Iv = compression_eigenvectors.dot(image.T)

    # Reconstruction approximative de l'image avec les vecteurs conservés
    Io = (compression_eigenvectors.T).dot(Iv).T

    return Io


def display_image_with_axes(image, title="Image"):
    """
    Affiche une image avec des axes montrant ses dimensions.

    Parameters:
    - image: numpy.ndarray, l'image à afficher.
    - title: str, titre de l'image affichée.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)


    # plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.show()

def main():
    image_original = np.load("image_complete.npy")
    image_sans_aberrations = fix_aberrration(image_original)
    show_and_compare_images(image_original, image_sans_aberrations, 'aberrations')

    rotated_image = rotate_image_90_right(image_sans_aberrations)
    show_and_compare_images(image_sans_aberrations, rotated_image, 'rotation')

    image_filtrer_bilineaire = filtre_methode_bilinieaire(rotated_image)
    show_and_compare_images(rotated_image, image_filtrer_bilineaire, 'bruit (bilineaire)')

    image_filtrer_elliptic = filtre_elliptic(rotated_image)
    show_and_compare_images(rotated_image, image_filtrer_elliptic, 'bruit (elliptic)')


    compressed_image_fifty = compress(image_filtrer_elliptic, factor=0.5)
    compressed_image_seventy = compress(image_filtrer_elliptic, factor=0.7)
    show_and_compare_images(image_filtrer_elliptic, compressed_image_fifty, 'pas de compression')
    display_image_with_axes(compressed_image_fifty, 'Image compresse 0.5')
    display_image_with_axes(compressed_image_seventy, 'Image compresse 0.7')

if __name__ == "__main__":
    main()