import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Charger l'image et la convertir en niveaux de gris
img = mpimg.imread('goldhill.png')

# Dimensions de l'image originale
original_height, original_width = img.shape

# Matrice de transformation
T = np.array([[2, 0], [0, 0.5]])

# Calcul des nouvelles dimensions
new_width = int(original_width * 2)  # Étirement par 2 en x
new_height = int(original_height * 0.5)  # Compression par 2 en y

# Créer une nouvelle image avec des zéros (noir) pour les pixels manquants
new_image = np.zeros((new_height, new_width))

# Inverse de la transformation pour faire correspondre les coordonnées à l'image originale
T_inv = np.linalg.inv(T)

# Appliquer la transformation inverse à chaque pixel de la nouvelle image
for i in range(new_height):
    for j in range(new_width):
        # Coordonnées inversées dans l'image originale
        original_coords = T_inv @ np.array([j, i])
        x, y = original_coords

        # Vérifier si les coordonnées sont dans les limites de l'image originale
        if 0 <= x < original_width and 0 <= y < original_height:
            # Remplir le pixel dans la nouvelle image
            new_image[i, j] = img[int(y), int(x)]

# Afficher l'image transformée
plt.figure(figsize=(10, 5))
plt.imshow(new_image, cmap='gray')
plt.title('Image étirée et écrasée')
plt.axis('off')
plt.show()
