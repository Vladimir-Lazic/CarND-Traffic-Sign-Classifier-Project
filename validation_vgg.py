from traffic_sign_classifier import *

# Suppressing TensorFlow FutureWarings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

label = y_train[index]

empty_image = np.zeros_like(image)
empty_label = 7

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./vgg_model/.'))
    model_prediction, prediction_index = predict_single_label(
        x_image=empty_image)
    print('Network Input: ', image_labels[label])
    print('Network Prediction: ', model_prediction)
    print('Prediction correct' if label ==
          prediction_index else 'Prediction incorrect')

    validation_accuracy = evaluate(X_valid, y_valid, model='vgg')
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
