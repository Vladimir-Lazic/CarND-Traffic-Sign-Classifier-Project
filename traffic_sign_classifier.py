# Load pickled data
from vggnet import *
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import warnings
from sklearn.utils import shuffle
import cv2

# Suppressing TensorFlow FutureWarings
with warnings.catch_warnings():
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


# Uncomment the network you want to use, and comment the one you don't
from lenet import *

training_file = './data_set/train.p'
validation_file = './data_set/valid.p'
testing_file = './data_set/test.p'

image_label_file = 'signnames.csv'


def parse_image_labels(input_csc_file):
    reader = csv.reader(open(input_csc_file, 'r'))
    retVal = {}
    for row in reader:
        key, value = row
        if key == 'ClassId':
            continue
        retVal.update({int(key): value})
    return retVal


def image_normalize(image):
    image = np.divide(image, 255)
    return image


def dataset_normalization(X_data):
    X_normalized = X_data.copy()
    num_examples = len(X_data)
    # print('Number of examples', num_examples)
    for i in range(num_examples):
        image = X_normalized[i]
        normalized_image = image_normalize(image)
        X_normalized[i] = normalized_image
    return X_normalized


# Parsing image label csv file
image_labels = parse_image_labels(image_label_file)

# Loading the data set
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert (len(X_train) == len(y_train))
assert (len(X_valid) == len(y_valid))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))

def dataset_grayscale(X_data):
    X_grayscale = []
    num_examples = len(X_data)
    for i in range(num_examples):
        image = X_data[i]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        X_grayscale.append(gray.reshape(32, 32, 1))
    return np.array(X_grayscale)

print('Grayscaling training set')
X_train = dataset_grayscale(X_train)
X_valid = dataset_grayscale(X_valid)
X_test = dataset_grayscale(X_test)

assert (len(X_train) == len(y_train))
print("Normalized Training Set:   {} samples".format(len(X_train)))

X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 150
BATCH_SIZE = 64

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, None)
one_hot_y = tf.one_hot(y, 43)

# Training pipeline
rate = 0.001

# Uncomment the network you want to use, and comment the one you don't
logits = LeNet(x)
# logits = VGGNet(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
# optimizer = tf.compat.v1.train.AdagradOptimizer(
#     rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad'
# )
training_operation = optimizer.minimize(loss_operation)

# Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data, model='lenet'):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + \
            BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        if model == 'lenet':
            accuracy = sess.run(
                accuracy_operation,
                feed_dict={
                    x: batch_x,
                    y: batch_y,
                    keep_prob_conv: 1.0,
                    keep_prob: 0.5})
        elif model == 'vgg':
            accuracy = sess.run(
                accuracy_operation,
                feed_dict={
                    x: batch_x,
                    y: batch_y,
                    keep_prob_conv: 1.0,
                    keep_prob: 1.0})

        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def predict_single_label(x_image):
    sess = tf.get_default_session()
    logits_output = sess.run(tf.argmax(logits, 1),
                             feed_dict={
                                 x: np.expand_dims(x_image, axis=0),
                                 keep_prob_conv: 1.0,
                                 keep_prob: 1.0})
    classification_index = logits_output[0]
    return image_labels[classification_index], classification_index


def batch_predict(X_data, BATCH_SIZE=64):
    num_examples = len(X_data)
    batch_predict = np.zeros(num_examples, dtype=np.int32)
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x = X_data[offset:offset + BATCH_SIZE]
        batch_predict[offset:offset + BATCH_SIZE] = sess.run(tf.argmax(
            logits, 1), feed_dict={x: batch_x, keep_prob_conv: 1.0, keep_prob: 0.5})
    return batch_predict
