import numpy as np
from typing import List
import matplotlib.pyplot as plt

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plots data to debug correct data format
def debug_data(train_images: np.array, train_labels: np.array, labels: List[str]):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[train_labels[i]])
    plt.show()

def plot_image(i: int, predictions_array: np.array, true_label: int, img: np.array):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:.2f}% ({})".format(labels[predicted_label],
                                100 * np.max(predictions_array),
                                labels[true_label]),
                                color=color)

def plot_value_array(i: int, predictions_array: np.array, true_label: int):
    iterable = range(10)
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(iterable)
    plt.yticks([])
    thisplot = plt.bar(iterable, predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

def plot_prediction(index: int, predictions: np.array, test_labels: np.array, test_images: np.array):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(index, predictions[index], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(index, predictions[index], test_labels)
    plt.show()

def print_accurracy(test_acc: float):
    print('Test accuracy: {:.2f}'.format(test_acc * 100))
