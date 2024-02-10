import numpy as np
from scipy.ndimage import gaussian_filter

def compute_hessian_matrix(image, sigma=1):
    """
    Calcule la matrice Hessienne d'une image à une échelle donnée sigma.

    :param image: Image d'entrée en niveaux de gris.
    :param sigma: Échelle à laquelle calculer la matrice Hessienne.
    :return: Les composantes de la matrice Hessienne (Hxx, Hxy, Hyy).
    """
    # Filtrage gaussien pour lisser l'image
    image_smoothed = gaussian_filter(image, sigma=sigma)

    # Calcul des dérivées secondes
    Hxx = gaussian_filter(image_smoothed, sigma=sigma, order=(2, 0))
    Hyy = gaussian_filter(image_smoothed, sigma=sigma, order=(0, 2))
    Hxy = gaussian_filter(image_smoothed, sigma=sigma, order=(1, 1))

    return Hxx, Hxy, Hyy

# Exemple d'utilisation
if __name__ == "__main__":
    from scipy import misc
    import matplotlib.pyplot as plt

    # Charger une image d'exemple et la convertir en niveaux de gris
    image = misc.face(gray=True)
    Hxx, Hxy, Hyy = compute_hessian_matrix(image, sigma=5)

    # Afficher les composantes de la matrice Hessienne
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Hxx, cmap='gray')
    plt.title('Hxx')
    plt.subplot(1, 3, 2)
    plt.imshow(Hxy, cmap='gray')
    plt.title('Hxy')
    plt.subplot(1, 3, 3)
    plt.imshow(Hyy, cmap='gray')
    plt.title('Hyy')
    plt.show()
