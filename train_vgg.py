import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.utils import shuffle
from traffic_sign_classifier import *

# Suppressing TensorFlow FutureWarings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    assert tf.test.is_gpu_available()
    assert tf.test.is_built_with_cuda()
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x,
                                                    y: batch_y,
                                                    keep_prob_conv: 1.0,
                                                    keep_prob: 0.5})

        print("EPOCH {} ...".format(i + 1))
        training_accuracy = evaluate(X_train, y_train, model='vgg')
        print("Training Accuracy = {:.3f}".format(training_accuracy))

        validation_accuracy = evaluate(X_valid, y_valid, model='vgg')
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './vgg_model/vgg')
    print("Model saved")
