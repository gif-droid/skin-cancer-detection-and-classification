
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:43:55 2026

@author: dap
"""

# importation des bibliotheque

import splitfolders
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16

import Augmentor

# divitsion de la base de donne en traon test et validation
"""
# data_dir = "/DATA"
# output_dir = "/DATA_SPLIT"

splitfolders.ratio(data_dir, output_dir, seed=1337, ratio=(.8, 0.1,0.1)) """

 

############################################################
# Augmentation de donné

# chemain d'acces pour le dossier d'entree et le dossier de sortie pour les donnees augmentees
input_d = r"C:\Users\ML\Desktop\bureau_etude\DATA"
output_d = r"C:\Users\ML\Desktop\bureau_etude\DATA_AUG"

#chemain pour le split
data_dir = r"C:\Users\ML\Desktop\bureau_etude\DATA_AUG"
output_dir = r"C:\Users\ML\Desktop\bureau_etude\DATA_AUG_SPLIT"

# creation d'un tableau avec le nom de chaque classe qui se trouve dans le dossier input

classe = ["actinic keratosis", "basal cell carcinoma", "dermatofibroma", 
          "melanoma", "nevus", "pigmented benign keratosis", "seborrheic keratosis",
          "squamous cell carcinoma", "vascular lesion"]

# boucle pour parcourir tous les classe et faire l'augmentation dessu

for i in classe:
    P = Augmentor.Pipeline(input_d + "\\" + i, output_d + "\\" + i)
    P.rotate(probability=1, max_left_rotation=5, max_right_rotation=10)
    
    # p.zoom(probability=0.8, min_factor=1.1, max_factor=1.5)
    
    # P.flip_left_right(probability=0.5)
    # P.flip_top_bottom(probability=0.5)

    # # Transformations colorimétriques
    # P.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)
    # P.random_color(probability=0.4, min_factor=0.9, max_factor=1.1)

    # # Crop aléatoire (excellent pour la robustesse) 
    # P.crop_random(probability=0.4, percentage_area=0.9)
    
    P.sample(1000)

splitfolders.ratio(data_dir, output_dir, seed=1337, ratio=(0.8, 0.1, 0.1))