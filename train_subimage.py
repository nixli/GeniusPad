import tensorflow as tf
import numpy as np
import os
import gzip
import tensorflow as tf
from tensorflow.python.platform import gfile
import atexit
import datetime

DATA = 0
LABEL = 1
NUMBER_OF_LABELS = 62

# parse error for reading data from file
class DataSetInvalidError(Exception):
    pass

# utility functions for reading and preparing datasets
# ideas taken from tensorflow mnist tutorials
def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def get_images(data):

    with gzip.GzipFile(fileobj=data) as bytes:
        magic = _read32(bytes)
        if magic != 2051:
            raise DataSetInvalidError("Wrong magic number")

        items = _read32(bytes)
        rows =  _read32(bytes)
        cols =  _read32(bytes)

        data_bytes =  bytes.read(rows*cols*items)
        image_data = np.frombuffer(data_bytes, dtype=np.uint8)
        image_data = image_data.reshape(items, rows* cols)

        return image_data


def get_labels(data, num_classes=NUMBER_OF_LABELS):

    with gzip.GzipFile(fileobj=data) as bytes:
        magic = _read32(bytes)
        if magic != 2049:
            raise DataSetInvalidError("Wrong magic number")

        items = _read32(bytes)
        data_bytes = bytes.read(items)

        labels = np.frombuffer(data_bytes, dtype=np.uint8)
        with tf.Session() as s:
            one_hot_labels = s.run(tf.one_hot(labels, num_classes))
        return one_hot_labels


def read_data(train, test):

    with gfile.Open(train[DATA], "rb") as data:
        train_data = get_images(data)

    with gfile.Open(train[LABEL], "rb") as label:
        train_label = get_labels(label)

    with gfile.Open(test[DATA], "rb") as data:
        test_data = get_images(data)

    with gfile.Open(test[LABEL], "rb") as label:
        test_label = get_labels(label)


    return (train_data, train_label), (test_data, test_label)


# returns a generator that produces data of size batch_size
def data_generator(data, batch_size=50, onepass=False):

    cur_batch_start = 0
    num_data = data[DATA].shape[0]
    assert (data[DATA].shape[0] == data[LABEL].shape[0])
    if batch_size > num_data:
        raise ValueError("Batch size {} is greater than data size {}".format(batch_size, num_data))
    while True:
        if cur_batch_start + batch_size <= num_data:

            yield (data[DATA][cur_batch_start: cur_batch_start + batch_size] > 0).astype(np.float32), \
                  data[LABEL][cur_batch_start:  cur_batch_start +batch_size].astype(np.float32)
            cur_batch_start += batch_size
        else:
            if onepass:
                raise StopIteration
            need = cur_batch_start + batch_size - num_data
            yield  (np.concatenate((data[DATA][cur_batch_start: num_data], data[DATA][0:need]), axis=0) >0).astype(np.float32), \
                   np.concatenate((data[LABEL][cur_batch_start: num_data], data[LABEL][0:need]), axis=0).astype(np.float32)
            cur_batch_start = need


# cnn layers to construct the network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def exit_save(s, x , y):
    inputs = {'x': x}
    outputs = {'y': y}
    date = str(datetime.datetime.now()).replace(" ", "_")
    tf.saved_model.simple_save(s, "./saved_model"+date, inputs, outputs)

def train():
        # use gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        ########################## load data from file ########################################
        train_data_path = [os.path.join(os.getcwd(), "AlphaNumericData", f) for f in
                 ("emnist-byclass-train-images-idx3-ubyte.gz",
                  "emnist-byclass-train-labels-idx1-ubyte.gz")]

        test_data_path = [os.path.join(os.getcwd(), "AlphaNumericData", f) for f in
                 ("emnist-byclass-test-images-idx3-ubyte.gz",
                  "emnist-byclass-test-labels-idx1-ubyte.gz")]

        train, test = read_data(train_data_path, test_data_path)
        #########################################################################################


        #################################### MODEL ###############################################
        # input layer
        x = tf.placeholder(tf.float32, [None, 784], name="input_x")

        # hidden layer 1
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # hidden layer 2
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # fully connected layer
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32, name="model_dropout")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, NUMBER_OF_LABELS])
        b_fc2 = bias_variable([NUMBER_OF_LABELS])

        # ouput layer
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # truth
        y_ = tf.placeholder(tf.float32, [None, NUMBER_OF_LABELS])

        # loss function
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        # optimizer
        train_step = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cross_entropy)

        answer = tf.argmax(y_conv, 1, name="output_y")
        correct_prediction = tf.equal(answer, tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #########################################################################################

        init = tf.initialize_all_variables()
        sess = tf.Session()
        atexit.register(exit_save, sess, x, y_)


        # prepare dataset
        train_data_length = train[0].shape[0]
        batch_size = 100
        num_epoches = 7
        num_iterations = num_epoches * train_data_length // batch_size + 1
        train_dataset = data_generator(train, batch_size=batch_size)
        test_dataset = data_generator(test, batch_size=batch_size, onepass=True)

        # start compute graph and train
        sess.run(init)
        epoch = 0
        print("\n\n\n" + "*" * 10 + " Optimising Model for {} epoches in {} iterations".format(num_epoches, num_iterations) + "*" * 10 + "\n\n\n")

        for i in range(num_iterations):
            if i* batch_size > train_data_length*epoch:
                epoch +=1
                print("\n\n\n" + "*" * 10 +" AT EPOCH {}".format(epoch) + "*" * 10 + "\n\n\n")

            batch_xs, batch_ys = next(train_dataset)

            if i % 500 == 0:

                train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                print("step %d, training accuracy %.3f" % (i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

        accu_sum = 0
        batch_count = 0
        for x_batch, y_batch in test_dataset:
            accu =  accuracy.eval(session=sess, feed_dict={x: x_batch, y_: y_batch, keep_prob: 1})
            accu_sum += accu
            batch_count +=1

        print("\n\n\n" + "*" * 10 + " FINAL ACCURACY: {}".format(accu_sum / batch_count) + "*" * 10 + "\n\n\n")



if __name__ == "__main__":
    train()