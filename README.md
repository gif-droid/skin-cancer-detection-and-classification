# Skin Cancer Classification using Deep Learning

## Introduction

## Context

This project was carried out as part of our **Master 1 EEA program**, specialization in **Signal, Image and Machine Learning**.

The course **“Bureau d’Étude – Apprentissage Automatique”** aims to introduce students to recent machine learning methods, with particular emphasis on **Deep Learning** using **Convolutional Neural Networks (CNNs)**.

Within this framework, we developed a model capable of assisting in the diagnosis of **skin cancer** from **dermoscopic images** of patients.

---

## Problem Statement

Skin cancer diagnosis is a common challenge in dermatology.

In many cases, it is difficult to visually distinguish between different lesion categories and accurately associate them with a specific cancer type because:

- lesions from different categories may appear very similar
- lesions belonging to the same category may present very different visual characteristics

Automating this classification process could help:

- improve diagnostic efficiency
- reduce human error
- reduce analysis time for dermatologists

---

## Project Objective

The objective of this project is twofold:

1. Design, train and validate a **Deep Learning model** capable of classifying skin lesions from dermoscopic images.
2. Use the predicted lesion class to:
   - identify **benign lesions** as **healthy**
   - identify **malignant lesions**
   - predict the **specific type of skin cancer** when malignancy is detected

---

## Model Deployment Logic

The project requirements specify that, once deployed, the model must be able to:

- determine whether the patient is **healthy**
- determine whether the patient has **skin cancer**
- specify the **type of cancer** if malignancy is detected

To achieve this, the dataset was analyzed and grouped into two major categories.

The dataset contains images distributed into **9 different lesion classes**, grouped as follows:

### Benign Lesions

These correspond to non-cancerous skin abnormalities:

- Dermatofibroma
- Nevus
- Pigmented Benign Keratosis
- Seborrheic Keratosis
- Vascular Lesion

### Malignant Lesions

These correspond to confirmed skin cancers:

- Actinic Keratosis
- Basal Cell Carcinoma
- Melanoma
- Squamous Cell Carcinoma

---

## Classification Strategy

The deployed system follows two steps.

### Step 1 — Lesion Screening

The model determines whether the lesion is:

- **Benign**
- **Malignant**

### Step 2 — Cancer Identification

If the lesion is malignant, the model predicts the exact cancer type.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Flask / FastAPI

---

## Example API Output

```json
{
  "diagnosis": "Malignant",
  "cancer_type": "Melanoma",
  "confidence": "91.3%"
}


Project Goal

The final goal of this project is to build an intelligent diagnostic assistance system capable of supporting dermatologists in the early detection of skin cancer.
