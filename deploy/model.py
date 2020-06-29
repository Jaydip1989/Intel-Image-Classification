import numpy as np 
import math
import keras
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam,rmsprop
from tensorflow.python.framework import ops 
ops.reset_default_graph()
import h5py

img_rows, img_cols = 32, 32
num_classes = 6
num_channels = 3 
batch_size = 150

train_data = "/users/dipit/Desktop/MyPython/Intel-Image-Classification/seg_train/"
val_data = "/users/dipit/Desktop/MyPython/Intel-Image-Classification/seg_test/"
test_data = '/users/dipit/Desktop/MyPython/Intel-Image-Classification/seg_pred'


train_datagen = ImageDataGenerator(rescale = 1./255,
								   height_shift_range = 0.3,
								   rotation_range = 45,
								   width_shift_range = 0.3,
								   fill_mode = "nearest",
								   zoom_range = 0.3,
								   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory=train_data,
													target_size = (img_rows, img_cols),
													class_mode = "categorical",
													shuffle = True,
													batch_size = batch_size)

validation_generator = validation_datagen.flow_from_directory(directory=val_data,
															target_size = (img_rows, img_cols),
															class_mode = "categorical",
															shuffle = False,
															batch_size = batch_size)

test_generator = test_datagen.flow_from_directory(directory = test_data,
												  target_size = (img_rows, img_cols),
												  class_mode = "categorical",
												  shuffle = False,
												  batch_size = batch_size)


model = Sequential()
model.add(Conv2D(32, (5,5), padding = "same", activation="relu",input_shape=(img_rows, img_cols, num_channels)))
model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides =2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation="softmax"))

print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer = Adam(lr=0.0001),
				metrics = ['accuracy'])

train_steps = math.ceil(train_generator.n/train_generator.batch_size)
val_steps = math.ceil(validation_generator.n/validation_generator.batch_size)
epochs = 150

history = model.fit(train_generator,
					epochs = epochs,
					steps_per_epoch = train_steps,
					validation_data = validation_generator,
					validation_steps = val_steps)

scores = model.evaluate_generator(validation_generator, steps=val_steps, verbose=1)
print("\n Validation Accuracy:%.3f Validation Loss:%.3f" %(scores[1]*100, scores[0]))

model.save_weights("model_100_epochs_adam.h5")

model_json = model.to_json()
with open("model_Intel_adam_1.json","w") as json_file:
	json_file.write(model_json)

	print("Model save to the disk")

val_predict = model.predict(validation_generator, steps=val_steps, verbose=1)
val_labels = np.argmax(val_predict)
print(val_predict)





















