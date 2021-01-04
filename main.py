import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from mpi4py import MPI


class Model:
    def __init__(self, lr):
        inputs = keras.Input(shape=(28, 28, 1), name="digits")
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPool2D()(conv1)
        conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPool2D()(conv2)
        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
        pool3 = layers.MaxPool2D()(conv3)
        flatten = layers.Flatten()(pool3)
        x1 = layers.Dense(64, activation="relu")(flatten)
        outputs = layers.Dense(10, name="predictions")(x1)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.optimizer = keras.optimizers.SGD(learning_rate=lr)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.shapes = []
        self.flat_gradient_shape = []
        self.calculate_gradients(x_train[0:1], y_train[0:1])

    def calculate_gradients(self, x_train, y_train):
        with tf.GradientTape() as tape:
            logits = self.model(x_train, training=True)
            loss_value = self.loss_fn(y_train, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        result = self.flatten_gradients(grads)
        self.flat_gradient_shape = result.numpy().shape
        return self.flatten_gradients(grads)

    def flatten_gradients(self, gradients):
        flat_grad = []
        shapes = []
        for arr in gradients:
            flat_grad.append(tf.reshape(arr, [-1, 1]))
            shapes.append(tf.shape(arr))
        self.shapes = shapes
        return tf.concat(flat_grad, axis=0)

    def update_params(self, flat_grad):
        output = []
        cntr = 0
        for shape in self.shapes:
            num_elements = tf.math.reduce_prod(shape)
            params = tf.reshape(flat_grad[cntr:cntr + num_elements, 0], shape)
            params = tf.cast(params, tf.float32)
            cntr += num_elements
            output.append(params)
        self.optimizer.apply_gradients(zip(output, self.model.trainable_weights))

    def report_performance(self):
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')
        test_idx = np.random.permutation(len(x_test))
        test_batch_idx = np.array_split(test_idx, 60)
        for batchIdx in test_batch_idx:
            logits = self.model(x_test[batchIdx], training=False)
            lossValue = self.loss_fn(y_test[batchIdx], logits)
            test_accuracy.update_state(y_test[batchIdx], logits)
            test_loss.update_state(lossValue)
        return test_accuracy.result().numpy(), test_loss.result().numpy()


class Job:
    def __init__(self, model_id, dataset_idx):
        self.model_id = model_id
        p1_len = int(len(dataset_idx)*x)
        dataset_idx1 = dataset_idx[0:p1_len]
        dataset_idx2 = dataset_idx[p1_len:]
        num_subparts_p1 = num_workers*(W-1)
        if len(dataset_idx1) % num_subparts_p1 != 0:
            missing = num_subparts_p1 - (len(dataset_idx1) % num_subparts_p1)
            to_be_added_idx = np.random.permutation(len(dataset_idx1))[0:missing]
            dataset_idx1 = np.concatenate((dataset_idx1, dataset_idx1[to_be_added_idx]))
        self.part1_pieces = np.array_split(dataset_idx1, num_subparts_p1)
        self.round = 0
    def get_minitask(self, worker_idx):
        if self.round < W-1:
            # uncoded tasks
            result = np.concatenate((np.array([0]), self.part1_pieces[self.round*num_workers + worker_idx]))
            return result
        else:
            # coded parts
            None





def master():
    job_queue = []
    model_under_operation = 0
    for slot in range(num_slots):
        # add new job to the queue
        idx = np.random.permutation(len(x_train))[0:num_workers*batch_size_per_worker]
        job_queue.append(Job(model_under_operation, idx))
        # transmit minitasks
        minitasks = [[] for _ in range(num_workers)]
        # update parameters in the workers

        for job in job_queue:
            for worker_idx in range(num_workers):
                minitasks[worker_idx].append(job.get_minitask(worker_idx))





def worker():
    None


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 28, 28, 1))/255.
x_test = np.reshape(x_test, (-1, 28, 28, 1))/255.
batch_size_per_worker = 256
num_slots = 100
num_workers = 4
W = 3
epsilon = 2
B = 2
x = (epsilon+1)*(W-1)/(B+W-1+epsilon*(W-1))

lr_list = [0.1, 0.2]
models = [Model(lr) for lr in lr_list]

# if rank == 0:
#     master()
#
# else:
#     worker()

