import numpy as np 
import keras
import tensorflow

from keras.optimizers import Adam
from keras.models import model_from_json
from tensorflow.python.framework import ops 
ops.reset_default_graph()
from PIL import Image
import PIL

json_file = open("/users/dipit/Desktop/MyPython/Intel-Image-Classification/deploy/model_Intel_adam_1.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("/users/dipit/Desktop/MyPython/Intel-Image-Classification/deploy/model_100_epochs_adam.h5")

loaded_model.compile(loss = "categorical_crossentropy",
					 optimizer = Adam(lr=0.0001),
					 metrics=['accuracy'])

IMG = Image.open('/users/dipit/Desktop/MyPython/Intel-Image-Classification/seg_pred/10013.jpg')
print(type(IMG))
IMG = IMG.resize((32,32))
IMG = np.array(IMG)
IMG = np.true_divide(IMG, 255)
IMG = IMG.reshape(1, 32, 32, 3)

predictions = loaded_model.predict(IMG)

predictions_c = loaded_model.predict_classes(IMG)
print(predictions, predictions_c)

print("\n")
print("\n")

classes = {
	'TRAIN':['BUILDINGS', 'FOREST', 'GLACIER', 'MOUNTAIN', 'SEA', 'STREET'],
	"VALIDATION":['BUILDINGS', 'FOREST', 'GLACIER', 'MOUNTAIN', 'SEA', 'STREET']
}

predicted_class = classes['TRAIN'][predictions_c[0]]
print("We think that is {}.".format(predicted_class.lower()))

