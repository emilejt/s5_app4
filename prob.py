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

    # Afficher les axes pour voir la taille de l'image
    plt.xlabel("Largeur (pixels)")
    plt.ylabel("Hauteur (pixels)")
    plt.colorbar(label='Intensité des pixels')

    # Garder les ticks pour voir les dimensions de l'image en pixels
    plt.xticks(np.arange(0, image.shape[1], step=max(1, image.shape[1] // 10)))
    plt.yticks(np.arange(0, image.shape[0], step=max(1, image.shape[0] // 10)))

    plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.show()

def main():
    image_aberre = np.load("goldhill_aberrations.npy")
    clean_image = fix_aberrration(image_aberre)
    #show_and_compare_images(image_aberre, clean_image, 'aberrations')

    image_bruite = np.load("goldhill_bruit.npy")
    image_filtrer_bilineaire = filtre_methode_bilinieaire(image_bruite)
    #show_and_compare_images(image_bruite, image_filtrer_bilineaire, 'bruit (bilineaire)')

    image_filtrer_elliptic = filtre_elliptic(image_bruite)
    #show_and_compare_images(image_bruite, image_filtrer_elliptic, 'bruit (elliptic)')
    plt.gray()
    image_a_tourner_couleur = mpimg.imread("goldhill_rotate.png")
    image_a_tourner = np.mean(image_a_tourner_couleur, -1)
    rotated_image = rotate_image_90_right(image_a_tourner)
    # show_and_compare_images(image_a_tourner, rotated_image, 'mauvaise rotation')

    compressed_image = compress(rotated_image, factor=0.5)
    compressed_image2 = compress(rotated_image, factor=0.7)
    # show_and_compare_images(rotated_image, compressed_image, 'compression')
    show_and_compare_images(compressed_image2, rotated_image, 'compression7')
    display_image_with_axes(rotated_image)
    display_image_with_axes(compressed_image)
    display_image_with_axes(compressed_image2)

if __name__ == "__main__":
    main()