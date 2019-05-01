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
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import np_utils
import os
import argparse
import json
#import PIL

# dimensions of our images.
#img_width, img_height = 150, 150

#top_model_weights_path = 'bottleneck_fc_model.h5'
#train_data_dir = 'data/train'
#validation_data_dir = 'data/validation'
#epochs = 50
#batch_size = 16
#number_of_classes = 3
#train_class_sizes = [10656, 8240, 10544]
# train_class1 = 10656
# train_class2 = 8240
# train_class3 = 10544
#validation_class_sizes  [3680, 2576, 3584]

# validation_class1 = 3680
# validation_class2 = 2576
# validation_class3 = 3584

#nb_train_samples = sum(train_class_sizes)
#nb_validation_samples = sum(validation_class_sizes)

def save_bottlebeck_features(train_data_dir, validation_data_dir, img_width, img_height, batch_size, features_path,nb_train_samples,nb_validation_samples):
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(os.path.join(features_path, 'bottleneck_features_train.npy'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(os.path.join(features_path, 'bottleneck_features_validation.npy'),bottleneck_features_validation)


def train_top_model(features_path, number_of_classes, top_model_weights_path, epochs, batch_size, train_class_sizes, validation_class_sizes):
    train_data = np.load(os.path.join(features_path, 'bottleneck_features_train.npy'))
    train_labels_list = []
    for i in range(number_of_classes):
        train_labels_list = train_labels_list + [i] * train_class_sizes[i]

    train_labels = np.array(train_labels_list)
    train_labels = np_utils.to_categorical(train_labels)

    validation_data = np.load(os.path.join(features_path, 'bottleneck_features_validation.npy'))
    validation_labels_list = []
    for i in range(number_of_classes):
        validation_labels_list = validation_labels_list + [i] * validation_class_sizes[i]

    validation_labels = np.array(validation_labels_list)
    validation_labels = np_utils.to_categorical(validation_labels)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


    
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--data_train', type=str, default='../data/train',
                        help='location of data where train folders are present')
    parser.add_argument('--data_val', type=str, default='../data/validation',
                        help='location of data where validation folders are present')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--model_weights_path', type=str, default='weights.pth',
                        help='path to save the top layer model')
    parser.add_argument('--features_path', type=str, default='../data/features/vgg_tensorflow/',
                        help='path to save the bottleneck features')
    parser.add_argument('--input_dim', type=int, default=150, help='input height and width of image')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
    

    # Parse arguments
    args = parser.parse_args()
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    
    #Hardcoded
    train_class_sizes = [64]*1000
    validation_class_sizes = [64]*1000
    nb_train_samples = sum(train_class_sizes)
    nb_validation_samples = sum(validation_class_sizes)
    
    save_bottlebeck_features(args.data_train, args.data_val, args.input_dim, args.input_dim, args.batch_size, args.features_path, nb_train_samples,nb_validation_samples)
    train_top_model(args.features_path, args.num_classes, args.model_weights_path, args.epochs, args.batch_size, train_class_sizes, validation_class_sizes)