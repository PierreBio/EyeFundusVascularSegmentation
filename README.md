# Automatic Vascular Segmentation of the eye
![logo_segmentation_small](https://github.com/PierreBio/EyeFundusVascularSegmentation/assets/45881846/b7d1d1cf-1a0a-47f5-87fb-55835557ee19)

This project is carried out in the context of the Artificial Intelligence Masters of **TelecomParis**.

<sub>Made with __Python v3.11__</sub>

## Project

Scanning Laser Ophthalmoscopy (SLO) is a retinal imaging modality that allows for the creation of a high-resolution, wide-field fundus image, enabling the observation of the majority of the retinal surface in a single image at a resolution between 10 and 100 μm. Besides diseases of the retina itself, observation of the fundus can diagnose several general pathologies by examining the arterial and venous circulation in the retina. This is particularly the case for arterial hypertension and renal insufficiency.

The diagnosis generally relies on a quantitative analysis of the entire vascular network of the retinal image, and thus requires precise segmentation of this network. The goal of this project is to propose an automatic method for segmenting the vascular network in SLO retinal images. Figure 1 shows two examples of SLO images, as well as the Ground Truth images, corresponding to the manual annotations of an expert.

![image](https://github.com/PierreBio/EyeFundusVascularSegmentation/assets/45881846/7ec2de94-321e-45c0-aea6-1454cf9ed9fc)

**Figure 1 – Two examples of SLO fundus images and expert annotations for segmentation of the vascular network.**

## Questions

1. Can you realize an optimized algorithm to segment the vascular network without using machine learning, nor deep learning? [answer](docs/ANSWERS.md#answer-1)
2. Can you explain the evaluation function provided in the "main" Python script? [answer](docs/ANSWERS.md#answer-2)
3. Why do we use two metrics (Precision and Recall)? [answer](docs/ANSWERS.md#answer-3)
4. What role does skeletonization play in this evaluation function? [answer](docs/ANSWERS.md#answer-4)

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

- Once the project is setup, you can launch it using our initial main script:

```
py -m src.main
```

- For our teacher, and to fill scholarship requirements, you can use this main_scholar script:

```
py -m src.main_scholar
```

## Method

See [our explanations](docs/METHOD.md) about the whole method inspired from [Wiharto, 2019](https://thesai.org/Downloads/Volume10No6/Paper_54-Blood_Vessels_Segmentation.pdf).

## Results

See [our results](docs/RESULTS.md).

## Ressources

1. [Frangi, 1998](https://d1wqtxts1xzle7.cloudfront.net/48947667/Muliscale_Vessel_Enhancement_Filtering20160918-18985-1pkzn9x-libre.pdf?1474259972=&response-content-disposition=inline%3B+filename%3DMuliscale_Vessel_Enhancement_Filtering.pdf&Expires=1707571192&Signature=Okg9BCjCipv9OxwQDCGuGKUJZt82qMVwROACadxa9fO0FvYchBxiPnXoXgmWerBwalYxut9hBc7pQ7gs93-PYfHBCjv3D-LGcdSTDSr0OjfK9E7yY2Im53wbuI6uc-kPkNxacXenQqeBKDDTEyb9WtlYrA1C2kPQoduNEL7VDg-smCTvoecvdiuz6V5g8Z8YCH8TBkQhU1zwwq-ntD~SRECvJ6nnOt6BhVvFvaP6Q7F7tb5fAwOhBh-DQqPwBcIyUi1sLpuAoBlTvagydDthv5AfDv~n8nzhqucy31IMZxMYAdH2rdBPCs7AjJfk7j2SDAU-WaEeu6v8khRjZum4zw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

2. [Wiharto, 2019](https://thesai.org/Downloads/Volume10No6/Paper_54-Blood_Vessels_Segmentation.pdf)
