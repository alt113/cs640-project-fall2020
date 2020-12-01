# let matplotlib plot the figures in the background
import matplotlib
matplotlib.use('Agg')


# import packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

PATH_TO_DIR = os.getcwd()

# parse CLI args
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to output serialized model')
ap.add_argument('-l', '--label-bin', required=True, help='path to output label binarizer')
ap.add_argument('-e', '--epochs', type=int, default=25, help='number of epochs')
ap.add_argument('-p', '--plot', type=str, default='plot.png', help='path to loss plot')
args = vars(ap.parse_args())

# lets initialize the labels and load the data after parsing our CLI
LABELS = set(['negative', 'neutral', 'positive'])

print('loading images...')

# imagePath = [paths.list_images(args['dataset'])]
imagePath = os.listdir(PATH_TO_DIR + '/' + args['dataset'])
imagePath.remove('.DS_Store')

data, labels = [], []
i = 0

# loop over the image paths
for img_path in imagePath:
    # extract the class label from the file name
    if img_path.find('positive') != -1:
        label = 'positive'
    elif img_path.find('negative') != -1:
        label = 'negative'
    else:
        label = 'neutral'
    # label = img_path.split(os.path.sep)[-2]

    # load the image
    image = cv2.imread(PATH_TO_DIR + '/' + args['dataset'] + '/' + img_path)

    # convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(img_path, image.shape, label, i)

    # fix image size to 224x224
    image = cv2.resize(image, (224, 224))

    data.append(image)
    labels.append(label)
    i += 1

# convert the data and labels into numpy arrays
data, labels = np.array(data), np.array(labels)

# one-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition data into train/test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# validation/testing data augmentation object
valAug = ImageDataGenerator()

# TODO - Look into these values
mean = np.array([123.68, 116.779, 103.939], dtype='float32')

trainAug.mean = mean
valAug.mean = mean

# load the ResNet50
baseModel = ResNet50(weights='imagenet',
                     include_top=False,
                     input_tensor=Input(shape=(224, 224, 3)))

# head of model that will go on top of baseModel
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# freeze weights on the base model so they do not get updated
for layer in baseModel.layers:
    layer.trainable = False

# start compiling + training process
print('compiling model...')
# TODO - Look into these values
opt = SGD(lr=1e-4, momentum=0.9, decay=(1e-4 / args['epochs']))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print('training model head...')
H = model.fit(
    x=trainAug.flow(trainX, trainY, batch_size=32),
    steps_per_epoch=len(trainX)//32,
    validation_data=valAug.flow(testX, testY),
    validation_steps=len(testX)//32,
    epochs=args['epochs']
)

# evaluation process
print('evaluating model...')
predictions = model.predict(x=testX.astype('float32'), batch_size=32)

print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))

# plot training loss/accuracy
N = args['epochs']
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='training loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='validation accuracy')
plt.title('Loss/Accuracy on Dataset')
plt.xlabel('Nbr of Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])

# wrap up - serialize model & label binarizer
print('serializing model to disc...')
model.save(args['model'], save_format='h5')

f = open(args['label_bin'], 'wb')
f.write(pickle.dumps(lb))
f.close()
