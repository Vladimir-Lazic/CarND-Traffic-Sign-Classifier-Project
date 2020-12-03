from traffic_sign_classifier import *

# Suppressing TensorFlow FutureWarings
with warnings.catch_warnings():
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

print("image: ", image.shape)

label = y_train[index]

empty_image = np.zeros_like(image)
empty_label = 7

saver = tf.train.Saver()




with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./model/.'))
    #model_prediction, prediction_index = predict_single_label(
    #    X_test[1])

    top_k_a = sess.run(top_k, feed_dict={x: np.expand_dims(X_test[1], axis=0), keep_prob_conv: 1.0, keep_prob:1.0})
    print(top_k_a)

