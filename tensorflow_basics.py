import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, losses
from utils import plot_value_array, plot_image, plot_prediction, print_accurracy

fashion_mnist = datasets.fashion_mnist

# Training data and testing data | Training set and dataset.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Labels to identify the indices in the train_labels list
# each train_label item is an index of this list.
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

probability_model = Sequential([model, layers.Softmax()])

predictions = probability_model.predict(test_images)

for i,j in enumerate(labels):
    plot_prediction(i, predictions, test_labels, test_images)
