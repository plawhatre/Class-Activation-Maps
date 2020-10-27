import tensorflow as tf
import numpy as np
import scipy as sp
import copy
from colorama import init
from termcolor import *
import cv2
init()
from glob import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
	# Load Data
	# images = glob('images/*.jp*g')
	images = glob('*.jp*g')
	# Load Model
	resnet = tf.keras.applications.ResNet50(
		input_shape=(224, 224, 3), weights='imagenet', include_top=True)

	# Layer before flatten
	activation_layer = resnet.get_layer(resnet.get_config()['layers'][-3]['name'])

	# Get weight matrix
	W = resnet.get_layer(resnet.get_config()['layers'][-1]['name']).get_weights()[0]

	# Create Model
	model = tf.keras.Model(inputs=resnet.input, outputs=activation_layer.output)

	for name in images:
		I = tf.keras.preprocessing.image.load_img(name, target_size=(224,224))

		x = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(I, axis=0))

		# Get feature maps
		fmaps = model.predict(x)[0]

		# Get Predictions
		prob = resnet.predict(x)
		cls_name_lst = tf.keras.applications.resnet50.decode_predictions(prob)[0]
		cprint(f'_______Probable class for image {name} _______\n{cls_name_lst}\n______________','green')
		y_pred = np.argmax(prob[0])
		label = cls_name_lst[0][1]

		# Get weight vector
		w = W[:, y_pred]

		CAM = fmaps.dot(w)
		# Zoom 32 times the size of the feature map
		CAM = sp.ndimage.zoom(CAM, (32,32))

		plt.figure()
		plt.subplot(121)
		plt.imshow(I, alpha=0.8)
		plt.imshow(CAM, cmap='jet', alpha=0.5)
		plt.subplot(122)
		plt.imshow(I)
		plt.title(label)
		filename = 'predcting_' + name
		plt.savefig(filename)
		plt.show()

		proceed = input('Do you want to continue?')
		if proceed.lower() == 'y':
			pass
		else:
			break