import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, InputLayer

#import dogbreedclasses

import warnings
warnings.filterwarnings("ignore") 

plt.rcParams['font.size'] = 10

from keras.models import load_model

classes = ['Chihuahua',
 'Japanese_spaniel',
 'Maltese_dog',
 'Pekinese',
 'Shih-Tzu',
 'Blenheim_spaniel',
 'Papillon',
 'Toy_terrier',
 'Rhodesian_ridgeback',
 'Afghan_hound',
 'Basset',
 'Beagle',
 'Bloodhound',
 'Bluetick',
 'Black-and-tan_coonhound',
 'Walker_hound',
 'English_foxhound',
 'Redbone',
 'Borzoi',
 'Irish_wolfhound',
 'Italian_greyhound',
 'Whippet',
 'Ibizan_hound',
 'Norwegian_elkhound',
 'Otterhound',
 'Saluki',
 'Scottish_deerhound',
 'Weimaraner',
 'Staffordshire_bullterrier',
 'American_Staffordshire_terrier',
 'Bedlington_terrier',
 'Border_terrier',
 'Kerry_blue_terrier',
 'Irish_terrier',
 'Norfolk_terrier',
 'Norwich_terrier',
 'Yorkshire_terrier',
 'Wire-haired_fox_terrier',
 'Lakeland_terrier',
 'Sealyham_terrier',
 'Airedale',
 'Cairn',
 'Australian_terrier',
 'Dandie_Dinmont',
 'Boston_bull',
 'Miniature_schnauzer',
 'Giant_schnauzer',
 'Standard_schnauzer',
 'Scotch_terrier',
 'Tibetan_terrier',
 'Silky_terrier',
 'Soft-coated_wheaten_terrier',
 'West_Highland_white_terrier',
 'Lhasa',
 'Flat-coated_retriever',
 'Curly-coated_retriever',
 'Golden_retriever',
 'Labrador_retriever',
 'Chesapeake_Bay_retriever',
 'German_short-haired_pointer',
 'Vizsla',
 'English_setter',
 'Irish_setter',
 'Gordon_setter',
 'Brittany_spaniel',
 'Clumber',
 'English_springer',
 'Welsh_springer_spaniel',
 'Cocker_spaniel',
 'Sussex_spaniel',
 'Irish_water_spaniel',
 'Kuvasz',
 'Schipperke',
 'Groenendael',
 'Malinois',
 'Briard',
 'Kelpie',
 'Komondor',
 'Old_English_sheepdog',
 'Shetland_sheepdog',
 'Collie',
 'Border_collie',
 'Bouvier_des_Flandres',
 'Rottweiler',
 'German_shepherd',
 'Doberman',
 'Miniature_pinscher',
 'Greater_Swiss_Mountain_dog',
 'Bernese_mountain_dog',
 'Appenzeller',
 'EntleBucher',
 'Boxer',
 'Bull_mastiff',
 'Tibetan_mastiff',
 'French_bulldog',
 'Great_Dane',
 'Saint_Bernard',
 'Eskimo_dog',
 'Malamute',
 'Siberian_husky',
 'Affenpinscher',
 'Basenji',
 'Pug',
 'Leonberg',
 'Newfoundland',
 'Great_Pyrenees',
 'Samoyed',
 'Pomeranian',
 'Chow',
 'Keeshond',
 'Brabancon_griffon',
 'Pembroke',
 'Cardigan',
 'Toy_poodle',
 'Miniature_poodle',
 'Standard_poodle',
 'Mexican_hairless',
 'Dingo',
 'Dhole',
 'African_hunting_dog']

model = tf.keras.models.load_model(
#      ('./model/model.h5'),
       ('model2.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

print(model.summary())

st.title('Please upload a picture of a dog to detect its breed')

uploaded_file = st.file_uploader("Upload your file here...")

if uploaded_file is not None:
	st.image(uploaded_file)
	from tensorflow.keras.preprocessing import image
	image1 = image.load_img(uploaded_file, target_size = (224, 224))

	transformedImage = image.img_to_array(image1)
	transformedImage = np.expand_dims(transformedImage, axis=0)
	prediction = model.predict(transformedImage)
	pred = np.argmax(prediction, axis=-1)

	st.write(f"The prediction of the dog breed is: {classes[pred[0]]}")