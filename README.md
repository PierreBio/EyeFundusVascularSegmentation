# Automatic Vascular Segmentation of the eye
Scanning Laser Ophthalmoscopy (SLO) is a retinal imaging modality that allows for the creation of a high-resolution, wide-field fundus image, enabling the observation of the majority of the retinal surface in a single image at a resolution between 10 and 100 μm. Besides diseases of the retina itself, observation of the fundus can diagnose several general pathologies by examining the arterial and venous circulation in the retina. This is particularly the case for arterial hypertension and renal insufficiency. 

The diagnosis generally relies on a quantitative analysis of the entire vascular network of the retinal image, and thus requires precise segmentation of this network. The goal of this project is to propose an automatic method for segmenting the vascular network in SLO retinal images. Figure 1 shows two examples of SLO images, as well as the Ground Truth images, corresponding to the manual annotations of an expert.

![image](https://github.com/PierreBio/EyeFundusVascularSegmentation/assets/45881846/7ec2de94-321e-45c0-aea6-1454cf9ed9fc)

**Figure 1 – Two examples of SLO fundus images and expert annotations for segmentation of the vascular network.**

## How to setup?

- First, clone the repository:

```
git clone https://github.com/PierreBio/EyeFundusVascularSegmentation.git
```

- Then go to the root of the project:

```
cd EyeFundusVascularSegmentation
```

- Create a virtual environment:

```
py -m venv venv
```

- Activate your environment:

```
.\venv\Scripts\activate
```

- Install requirements:

```
pip install -r requirements.txt
```

## How to launch?

- Once the project is setup, you can launch it:

```
py -m src.main
```

## Explained Method


## Results


## Ressources
