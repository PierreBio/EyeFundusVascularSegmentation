def compute_vesselness(Hxx, Hxy, Hyy, beta=0.5, c=15):
    """
    Calcule la "vesselness" pour chaque point de l'image en utilisant les valeurs propres de la matrice Hessienne.

    :param Hxx: Dérivée seconde de l'image par rapport à x.
    :param Hxy: Dérivée seconde mixte de l'image.
    :param Hyy: Dérivée seconde de l'image par rapport à y.
    :param beta: Paramètre de sensibilité pour la luminosité des vaisseaux.
    :param c: Paramètre de seuillage pour filtrer les zones de faible contraste.
    :return: Image de la "vesselness".
    """
    # Calcul des valeurs propres
    D = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
    lambda1 = 0.5*(Hxx + Hyy + D)
    lambda2 = 0.5*(Hxx + Hyy - D)

    # Assurer que |lambda1| <= |lambda2|
    lambda1, lambda2 = np.where(np.abs(lambda1) > np.abs(lambda2), lambda2, lambda1), np.where(np.abs(lambda1) > np.abs(lambda2), lambda1, lambda2)

    # Calcul de la "vesselness"
    Rb = (lambda1 / lambda2)**2
    S2 = lambda1**2 + lambda2**2
    V = np.exp(-Rb / (2*beta**2)) * (1 - np.exp(-S2 / (2*c**2)))

    # Eliminer les valeurs propres négatives (qui correspondent à des structures en forme de plaque)
    V[lambda2 > 0] = 0

    return V

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

def frangi_filter_response(image, sigmas, beta=0.5, c=15):
    """
    Calcule la réponse du filtre de Frangi pour chaque point de l'image à différentes échelles.

    :param image: Image d'entrée en niveaux de gris.
    :param sigmas: Liste des échelles (sigma) à utiliser pour le calcul.
    :param beta: Paramètre de la formule de Frangi.
    :param c: Paramètre de la formule de Frangi.
    :return: Image de la réponse du filtre de Frangi.
    """
    vesselness_images = []
    for sigma in sigmas:
        Hxx, Hxy, Hyy = compute_hessian_matrix(image, sigma)
        V = compute_vesselness(Hxx, Hxy, Hyy, beta, c)
        vesselness_images.append(V)

    # Sélectionner la valeur maximale de vesselness pour chaque point à travers les échelles
    frangi_response = np.max(vesselness_images, axis=0)

    return frangi_response

def test_frangi_parameters(image, sigmas_list, beta_list, c_list):
    """
    Teste différentes combinaisons de paramètres pour l'algorithme de Frangi et affiche les résultats.

    :param image: Image d'entrée en niveaux de gris.
    :param sigmas_list: Liste des ensembles de sigmas à tester.
    :param beta_list: Liste des valeurs de beta à tester.
    :param c_list: Liste des valeurs de c à tester.
    """
    fig, axs = plt.subplots(len(sigmas_list), len(beta_list) * len(c_list), figsize=(20, 10))

    for i, sigmas in enumerate(sigmas_list):
        for j, beta in enumerate(beta_list):
            for k, c in enumerate(c_list):
                frangi_response = frangi_filter_response(image, sigmas, beta, c)
                ax = axs[i, j * len(c_list) + k]
                ax.imshow(frangi_response, cmap='gray')
                ax.set_title(f'Sigmas: {sigmas}\nBeta: {beta}, C: {c}')
                ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt
    import imageio
    from skimage.color import rgb2gray

    # Charger l'image téléchargée
    image = imageio.imread('data/images_IOSTAR/star01_OSC.jpg')
    if image.ndim == 3:
        image = rgb2gray(image)

    # Définir les ensembles de paramètres à tester
    sigmas_list = [[0.5, 1, 1.5, 2], [0.6, 0.8, 1, 1.2]]
    beta_list = [0.5, 0.75]
    c_list = [15, 10]

    # Tester les paramètres
    test_frangi_parameters(image, sigmas_list, beta_list, c_list)