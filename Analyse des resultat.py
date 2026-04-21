# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:51:47 2026

@author: ML
"""


import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16

from keras.models import Model
import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib.image as mpimg
%matplotlib inline
import os

# bibliothèque pour importer des images
from PIL import Image
import numpy as np
import cv2
from keras.models import Model

!pip install split-folders
!pip install Augmentor
import Augmentor
import splitfolders
# test de l'architecture sauvegarder
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_architecture = 'model.json'
model_weights = 'model.h5'

model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

model.summary()
model.compile(optimizer='SGD', loss='categorical_crossentropy',metrics=['accuracy'])

#### pretraitement des donnees
datagen= ImageDataGenerator(rescale=1./255)


test_generator=datagen.flow_from_directory(
    r"C:\Users\ML\Desktop\bureau_etude\DATA_AUG_SPLIT\test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    seed=42
    )



#evaluation du nouveau model
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)



## Visualisation des filtres et des caractéristiques extraites par le réseau
model.summary()

base_model = model.layers[0]
base_model.summary()

# Récupérer les filtres et les poids de la 1ère couche

filters, biases, *is_anything_else_being_returned = base_model.layers[1].get_weights()
# normaliser les filtres sur [0 , 1]
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# Prendre le deuxième filtre de cette couche (qui contient 3 sous-filtres pour les 3 canaux R,
# V et et B) et visualiser ces 3 sous-filtres
f=filters[:,:,:,1]
plt.figure()
plt.subplot(1,3,1)
plt.imshow(f[:, :, 0], cmap='gray')
plt.subplot(1,3,2)
plt.imshow(f[:, :, 1], cmap='gray')
plt.subplot(1,3,3)
plt.imshow(f[:, :, 2], cmap='gray')


# Maintenant, on veut prendre une image et la mettre en entrée d’un seul filtre et voir ce qui est donné en sortie

# On importe l’image et on lui applique toutes les transformations que l’on a appliquées aux images pendant l’entraînement

img1 ='/content/drive/MyDrive/Colab Notebooks/DATA/Train/actinic keratosis/ISIC_0031430.jpg'
img2 ='/content/drive/MyDrive/Colab Notebooks/DATA_AUG_SPLIT/test/actinic keratosis/actinic keratosis_original_ISIC_0026650.jpg_0daf35c3-7b06-44ab-9af9-6cff447c6c26.jpg'

IMG1 = Image.open(img1)
IMG2 = Image.open(img2)
# pretraitement sur l'image1
image1 = np.asarray(IMG1)
image1=image1.astype('float32')
image1=cv2.resize(image1,(224,224))
image1 /= 255
image1 = image1.reshape([-1,224,224,3])

# pretraitement sur l'image2
image2 = np.asarray(IMG2)
image2=image2.astype('float32')
image2=cv2.resize(image2,(224,224))
image2 /= 255
image2 = image2.reshape([-1,224,224,3])

########### Définir un modèle intermédiaire contenant les 2 premières couches du modèle initial
inter_model = Model(inputs=base_model.inputs, outputs=base_model.layers[1].output)
inter_model.summary()
feature_maps = inter_model.predict(image1)

plt.figure()
plt.subplot(121)
plt.imshow(IMG1)
plt.subplot(122)
plt.imshow(feature_maps[0,:,:,0])

## Même chose pour l’image 2
inter_model = Model(inputs=base_model.inputs, outputs=base_model.layers[2].output)
inter_model.summary()
feature_maps = inter_model.predict(image2)

plt.figure()
plt.subplot(121)
plt.imshow(IMG2)
plt.subplot(122)
plt.imshow(feature_maps[0,:,:,0])




from sklearn.metrics import confusion_matrix
test_images = test_generator


y_pred_prob=model.predict(test_images)
from sklearn.metrics import confusion_matrix
test_images = test_generator


y_pred_prob=model.predict(test_images)

y_pred=np.argmax(y_pred_prob, 1)

labels = test_images.classes

mat = confusion_matrix(labels, y_pred)

# Affichage
plt.figure()
sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='black')

plt.title("Matrice de confusion")
plt.xlabel("Valeurs prédites")
plt.ylabel("Valeurs réelles")

plt.show()

# Affichage des metriques

from sklearn.metrics import classification_report

print(classification_report(labels, y_pred))
