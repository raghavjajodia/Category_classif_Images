'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created class1/ and class2/ class3/ subfolders inside train/ and validation/
- put the class1 pictures index 0-999 in data/train/class1
- put the class1 pictures index 1000-1400 in data/validation/class1
- put the class2 pictures index 12500-13499 in data/train/class2
- put the class2 pictures index 13500-13900 in data/validation/class2
- put the class3 pictures index 12500-13499 in data/train/class3
- put the class3 pictures index 13500-13900 in data/validation/class3
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        class1/
            class1001.jpg
            class1002.jpg
            ...
        class2/
            class2001.jpg
            class2002.jpg
            ...
    validation/
        class1/
            class1001.jpg
            class1002.jpg
            ...
        class2/
            class2001.jpg
            class2002.jpg
            ...
```
'''

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input

# path to the model weights files.
#VGG16 weights path
weights_path = '../keras/examples/vgg16_weights.h5'
#top layer weights path trained by transfer learning
top_model_weights_path = 'fc_model.h5'
# fine tuned weights output-path
fineTune_output_path = 'final_model.h5'

# dimensions of our images.
img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16
num_classes = 3

input_tensor = Input(shape=(150,150,3))
# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

model = Model(input= base_model.input, output= top_model(base_model.output))
# add the model on top of the convolutional base
#model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None)

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model.save_weights(fineTune_output_path)