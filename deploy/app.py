from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

import numpy as np 
import PIL
from PIL import Image
import os


app = Flask(__name__)

MODEL_ARCHITECTURE = "/users/dipit/Desktop/MyPython/Intel-Image-Classification/deploy/model_Intel_adam_1.json"
MODEL_WEIGHTS= "/users/dipit/Desktop/MyPython/Intel-Image-Classification/deploy/model_100_epochs_adam.h5"

json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(MODEL_WEIGHTS)
print(" Model Loaded.Check http://127.0.0.1:5000/")

def model_predict(img_path, model):

	IMG = image.load_img(img_path, target_size=(32,32))
	print(type(IMG))

	IMG = IMG.resize((32,32))
	IMG = np.array(IMG)
	IMG = np.true_divide(IMG, 255)
	IMG = IMG.reshape(1, 32, 32, 3)
	#IMG = np.expand_dims(IMG, axis=0)

	print(model)

	model.compile(loss = "categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])
	prediction = model.predict_classes(IMG)

	return prediction


@app.route("/", methods=["GET"])
def index():
	# Main Page
	return render_template('index.html')


@app.route("/predict",methods=['GET', "POST"])
def upload():

	classes = {
			'TRAIN':['BUILDINGS', 'FOREST', 'GLACIER', 'MOUNTAIN', 'SEA', 'STREET'],
			"VALIDATION":['BUILDINGS', 'FOREST', 'GLACIER', 'MOUNTAIN', 'SEA', 'STREET']}

	if request.method == "POST":

		# get the file from post request
		f = request.files['file']

		# save the file to uploads

		basepath = os.path.dirname(__file__)

		file_path = os.path.join(
			basepath, "uploads", secure_filename(f.filename))
		f.save(file_path)

		prediction = model_predict(file_path, model)

		predicted_class = classes['VALIDATION'][prediction[0]]
		print("We think that is {} .".format(predicted_class.lower()))

		return str(predicted_class).lower()


if __name__ == "__main__":
	app.run(debug = True)





















