import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

keep_prob_conv = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)


def VGG(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for
    # the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1 (Convolutional): Input = 32x32x1. Output = 32x32x32.
    conv1_W = tf.Variable(
        tf.truncated_normal(
            shape=(
                3,
                3,
                3,
                32),
            mean=mu,
            stddev=sigma),
        name='conv1_W')
    conv1_b = tf.Variable(tf.zeros(32), name='conv1_b')
    conv1 = tf.nn.conv2d(
        x, conv1_W, strides=[
            1, 1, 1, 1], padding='SAME') + conv1_b

    # ReLu Activation.
    conv1 = tf.nn.relu(conv1)

    # Layer 2 (Convolutional): Input = 32x32x32. Output = 32x32x32.
    conv2_W = tf.Variable(
        tf.truncated_normal(
            shape=(
                3,
                3,
                32,
                32),
            mean=mu,
            stddev=sigma),
        name='conv2_W')
    conv2_b = tf.Variable(tf.zeros(32), name='conv2_b')
    conv2 = tf.nn.conv2d(
        conv1, conv2_W, strides=[
            1, 1, 1, 1], padding='SAME') + conv2_b

    # ReLu Activation.
    conv2 = tf.nn.relu(conv2)

    # Layer 3 (Pooling): Input = 32x32x32. Output = 16x16x32.
    conv2 = tf.nn.max_pool(
        conv2, ksize=[
            1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.dropout(conv2, keep_prob_conv)

    # Layer 4 (Convolutional): Input = 16x16x32. Output = 16x16x64.
    conv3_W = tf.Variable(
        tf.truncated_normal(
            shape=(
                3,
                3,
                32,
                64),
            mean=mu,
            stddev=sigma),
        name='conv3_W')
    conv3_b = tf.Variable(tf.zeros(64), name='conv3_b')
    conv3 = tf.nn.conv2d(
        conv2, conv3_W, strides=[
            1, 1, 1, 1], padding='SAME') + conv3_b

    # ReLu Activation.
    conv3 = tf.nn.relu(conv3)

    # Layer 5 (Convolutional): Input = 16x16x64. Output = 16x16x64.
    conv4_W = tf.Variable(
        tf.truncated_normal(
            shape=(
                3,
                3,
                64,
                64),
            mean=mu,
            stddev=sigma),
        name='conv4_W')
    conv4_b = tf.Variable(tf.zeros(64), name='conv4_b')
    conv4 = tf.nn.conv2d(
        conv3, conv4_W, strides=[
            1, 1, 1, 1], padding='SAME') + conv4_b

    # ReLu Activation.
    conv4 = tf.nn.relu(conv4)

    # Layer 6 (Pooling): Input = 16x16x64. Output = 8x8x64.
    conv4 = tf.nn.max_pool(
        conv4, ksize=[
            1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='VALID')
    conv4 = tf.nn.dropout(conv4, keep_prob_conv)  # dropout

    # Layer 7 (Convolutional): Input = 8x8x64. Output = 8x8x128.
    conv5_W = tf.Variable(
        tf.truncated_normal(
            shape=(
                3,
                3,
                64,
                128),
            mean=mu,
            stddev=sigma),
        name='conv5_W')
    conv5_b = tf.Variable(tf.zeros(128), name='conv5_b')
    conv5 = tf.nn.conv2d(
        conv4, conv5_W, strides=[
            1, 1, 1, 1], padding='SAME') + conv5_b

    # ReLu Activation.
    conv5 = tf.nn.relu(conv5)

    # Layer 8 (Convolutional): Input = 8x8x128. Output = 8x8x128.
    conv6_W = tf.Variable(
        tf.truncated_normal(
            shape=(
                3,
                3,
                128,
                128),
            mean=mu,
            stddev=sigma),
        name='conv6_W')
    conv6_b = tf.Variable(tf.zeros(128), name='conv6_b')
    conv6 = tf.nn.conv2d(
        conv5, conv6_W, strides=[
            1, 1, 1, 1], padding='SAME') + conv6_b

    # ReLu Activation.
    conv6 = tf.nn.relu(conv6)

    # Layer 9 (Pooling): Input = 8x8x128. Output = 4x4x128.
    conv6 = tf.nn.max_pool(
        conv6, ksize=[
            1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='VALID')
    conv6 = tf.nn.dropout(conv6, keep_prob_conv)  # dropout

    # Flatten. Input = 4x4x128. Output = 2048.
    fc0 = tf.layers.flatten(conv6)

    # Layer 10 (Fully Connected): Input = 2048. Output = 128.
    fc1_W = tf.Variable(
        tf.truncated_normal(
            shape=(
                2048,
                128),
            mean=mu,
            stddev=sigma),
        name='fc1_W')
    fc1_b = tf.Variable(tf.zeros(128), name='fc1_b')
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # ReLu Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)  # dropout

    # Layer 11 (Fully Connected): Input = 128. Output = 128.
    fc2_W = tf.Variable(
        tf.truncated_normal(
            shape=(
                128,
                128),
            mean=mu,
            stddev=sigma),
        name='fc2_W')
    fc2_b = tf.Variable(tf.zeros(128), name='fc2_b')
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # ReLu Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)  # dropout

    # Layer 12 (Fully Connected): Input = 128. Output = 43.
    fc3_W = tf.Variable(
        tf.truncated_normal(
            shape=(
                128,
                43),
            mean=mu,
            stddev=sigma),
        name='fc3_W')
    fc3_b = tf.Variable(tf.zeros(43), name='fc3_b')
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
