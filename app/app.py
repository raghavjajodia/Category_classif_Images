from keras import applications
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
import numpy as np
from flask import Flask, render_template, request, url_for
import requests
import re
import numpy as np
import operator
import pandas as pd
from collections import OrderedDict
import os
import simplejson as json
import datetime
import sys
from PIL import Image, ImageOps
import uuid

# path to the model weights files.
top_model_weights_path = '../bottleneck_fc_model.h5'
# dimensions of our images. This is set according to the input dimension of the model
img_width, img_height = 150, 150
# this is set in accordance with model output
num_classes = 3

#initialize and load trained model
input_tensor = Input(shape=(img_height,img_width,3))

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)
print('Base Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# load the weights of top model
top_model.load_weights(top_model_weights_path)
print('Top Model loaded')

# Compile the full model by adding topmodel to base model
model = Model(input= base_model.input, output= top_model(base_model.output))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
print('Full Model loaded')

datagen = ImageDataGenerator(rescale=1. / 255)
#loading complete

app = Flask(__name__)

#function to resize the incoming image and store it in file path
#This does not crop the image
#It adds black background to make the image as square and centers the image
#returns path of the resized image
def resize(fullPath, fileName, newHeight, newWidth, newImageFolder):
    im = Image.open(fullPath)
    w, h  = im.size

    if(w>h):
        nw = w
        nh = w
        deltah = nh-h
        ltrb_border = (0,deltah//2, 0, deltah - (deltah//2) )
    else:
        nw = h
        nh= h
        deltaw = nw-w
        ltrb_border = (deltaw//2, 0, deltaw - (deltaw//2), 0)

    newIm = ImageOps.expand(im, border = ltrb_border, fill = 'black')
    newIm = newIm.resize((newWidth, newHeight), Image.ANTIALIAS)
    extension = im.format.lower()
    fileName = os.path.splitext(fileName)[0]
    newImageFullPath = os.path.join(newImageFolder, fileName + "." + extension)
    newIm.save(newImageFullPath)
    return newImageFullPath

#returns list of probabilities for each class
def cecClassPrediction(fileName):
	print("Predicting: " + fileName)
	img = load_img(fileName)
	testImage = img_to_array(img)
	testImage = np.expand_dims(testImage, axis=0)

	prediction = ""

	for someImage in datagen.flow(testImage, batch_size=1):
	    prediction = model.predict(someImage)
	    break

	prediction = prediction[0].tolist()
	return prediction

#End point to report incorrectly predicted value and save this in a file
#End point format /ReportIncorrect?traceId=slfjalsfdjl&correctLabel=1
@app.route("/ReportIncorrect", methods=['GET','POST'])
def report_incorrect():
	traceId = request.args.get("traceId")
	correctValue = request.args.get("correctLabel")
	incorrectData = "static/incorrect_data/incorrectData.tsv"

	# Save to output file
	with open(incorrectData, "a+") as result_file:
		result_file.write("%s\t%s\n" % (traceId, correctValue))

	return json.dumps({'Result': "Successfully Recorded"})

#End point to predict category
#Image is sent as binary data in POST request on URL /CECCategoryClassifier
@app.route("/CECCategoryClassifier", methods=['GET','POST'])
def predict_cecCategory():
	if request.method == 'POST':
		print("getting file from the request")
		request.get_data()
		imageData = request.data
		currentDateTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
		fileName = currentDateTime + ".jpg"
		
		uploadFolder = "static/uploaded_data/"
		resizedFolder =  "static/resized_data/"
		predictedResult = "static/predicted_output/results.tsv"

		fname = uploadFolder + fileName
		with open(fname, 'wb+') as f:
			f.write(imageData)		
		finalImagePath = resize(fname, fileName, img_width, img_height, resizedFolder)
		print("finalImagePath : " + finalImagePath)
		probabilities = cecClassPrediction(finalImagePath)
		print("got prediction: ")
		print(probabilities)

		#Generate a traceId
		traceId = str(uuid.uuid4())

		# Save to output file
		with open(predictedResult, "a") as result_file:
			result_file.write("%s\t%s\t%s\t%s\t%s\n" % (currentDateTime, traceId, fname, finalImagePath, ','.join(probabilities)))
		return json.dumps({'Timestamp': currentDateTime, 'TraceId': traceId , "Probabilities": ', '.join(probabilities)})

if __name__ == "__main__":
	app.run(host='0.0.0.0')