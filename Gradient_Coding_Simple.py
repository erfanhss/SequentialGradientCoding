import time
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
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        self.shapes = []
        self.flat_gradient_shape = []
        self.calculate_gradients([x_train[0:1]], [y_train[0:1]], [1])
        self.accuracy = []
        self.loss = []

    def calculate_gradients(self, x_train, y_train, coefficients):
        with tf.GradientTape() as tape:
            loss_value = 0
            for (x_train_piece, y_train_pieces, coef) in zip(x_train, y_train, coefficients):
                logits = self.model(x_train_piece, training=True)
                loss_value += self.loss_fn(y_train_pieces, logits) * coef
        grads = tape.gradient(loss_value, self.model.trainable_weights, )
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

    def unflatten(self, flat_grad):
        output = []
        cntr = 0
        for shape in self.shapes:
            num_elements = tf.math.reduce_prod(shape)
            params = tf.reshape(flat_grad[cntr:cntr + num_elements, 0], shape)
            params = tf.cast(params, tf.float32)
            cntr += num_elements
            output.append(params)
        return output

    def update_params(self, flat_grad):
        output = self.unflatten(flat_grad)
        self.optimizer.apply_gradients(zip(output, self.model.trainable_weights))
        acc, loss = self.report_performance()
        self.accuracy.append(acc)
        self.loss.append(loss)

    def report_performance(self):
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')
        test_idx = np.random.permutation(len(x_test))
        test_batch_idx = np.array_split(test_idx, 60)
        for batchIdx in test_batch_idx:
            logits = self.model(x_test[batchIdx], training=False)
            lossValue = self.loss_fn(y_test[batchIdx], logits)/len(batchIdx)
            test_accuracy.update_state(y_test[batchIdx], logits)
            test_loss.update_state(lossValue)
        return test_accuracy.result().numpy(), test_loss.result().numpy()


def divide_into_pieces(array, num_parts):
    tmp = np.copy(array)
    if len(array) % num_parts != 0:
        missing = num_parts - (len(array) % num_parts)
        to_be_added_idx = np.random.permutation(len(array))[0:missing]
        tmp = np.concatenate((array, array[to_be_added_idx]))
    return np.array_split(tmp, num_parts)


def master():
    model_under_operation = 0
    time_spent = np.zeros([num_workers, num_slots * 2], dtype=float)
    identifiers = np.arange(num_workers)
    piece_map = [[(worker_idx + i) % num_workers for i in range(epsilon + 1)] for worker_idx in
                      range(num_workers)]
    complement_piece_map = [[(worker_idx + 1 + i) % num_workers for i in range(num_workers - epsilon - 1)] for
                            worker_idx
                            in range(num_workers)]
    coefficients = [
        [np.prod([identifiers[worker_idx] - identifiers[j] for j in complement_piece_map[idx]]) for idx in
         piece_map[worker_idx]] for worker_idx in range(num_workers)]
    round_times = []
    for slot in range(num_slots):
        crt_model = models[model_under_operation]
        print(slot)
        # transmit model parameters
        params = crt_model.flatten_gradients(crt_model.model.get_weights())
        param_req = []
        for worker_idx in range(num_workers):
            param_req.append(comm.Isend(np.ascontiguousarray(params, dtype=float), dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(param_req)
        # add new job to the queue
        idx = np.random.permutation(len(x_train))[0:num_workers * batch_size_per_worker]
        idx_parts = divide_into_pieces(idx, num_workers)
        worker_tasks = [[idx_parts[i] for i in piece_map[worker_idx]] for worker_idx in range(num_workers)]
        # transmit number of subparts and length of each
        reqs = []
        for worker_idx in range(num_workers):
            reqs.append(comm.Isend(np.array([len(worker_tasks[worker_idx]), len(worker_tasks[worker_idx][0])]),
                                   dest=worker_idx+1, tag=0))
        MPI.Request.waitall(reqs)
        # transmit tasks
        reqs = []
        for worker_idx in range(num_workers):
            for subtask in worker_tasks[worker_idx]:
                reqs.append(comm.Isend(np.ascontiguousarray(subtask, dtype=int), dest=worker_idx+1, tag=0))
            reqs.append(comm.Isend(np.ascontiguousarray(coefficients[worker_idx], dtype=float), dest=worker_idx+1, tag=1))
        MPI.Request.waitall(reqs)
        # receive results

        results = [np.zeros(crt_model.flat_gradient_shape, dtype=float) for _ in range(num_workers)]
        reqs = []
        for worker_idx, res in enumerate(results):
            reqs.append(comm.Irecv(res, source=worker_idx+1, tag=0))
        MPI.Request.waitall(reqs)
        # print(results)
        for worker_idx in range(num_workers):
            time_spent[worker_idx, slot] = comm.recv(source=worker_idx+1, tag=0)

        # identify stragglers
        crt_round_times = time_spent[:, slot]
        sorted_idx = np.argsort(crt_round_times)
        rec_idx = sorted_idx[0:num_workers-epsilon]
        round_times.append(crt_round_times[sorted_idx[num_workers-epsilon-1]])
        # compute the update
        pieces = [results[i] for i in rec_idx]
        rec_ident = [identifiers[i] for i in rec_idx]
        vandermonde_matrix = np.array([[j ** i for j in rec_ident] for i in range(num_workers - epsilon)])
        inv_vand_mat = np.linalg.inv(vandermonde_matrix)
        recon_coef = inv_vand_mat[:, -1]
        recon = np.sum([recon_coef[i] * pieces[i] for i in range(num_workers - epsilon)], axis=0)
        crt_model.update_params(recon/len(np.concatenate(idx_parts)))
        model_under_operation = (model_under_operation + 1) % len(models)
    for idx, model in enumerate(models):
        print(model.report_performance())
        np.save('Model_' + str(idx) + 'grad_code_test_loss', model.loss)
        np.save('Model_' + str(idx) + 'grad_code_test_accuracy', model.accuracy)
    np.save('round_times_grad_coding', np.array(round_times))
    # print(round_times)
def worker():
    model_under_operation = 0
    state = 0
    seed_arr = np.random.RandomState(seed=rank+2).randint(0, 100000, size=num_slots)
    for slot in range(num_slots):
        # determine whether a node is straggler
        if state == 0:
            straggling_status = 0
        else:
            straggling_status = 1
        if state == 0:
            if np.random.RandomState(seed=seed_arr[slot]).binomial(1, a):
                state = 1
        else:
            if np.random.RandomState(seed=seed_arr[slot]).binomial(1, b):
                state = (state + 1) % (num_states + 1)
        # receive new parameters from master
        crt_model = models[model_under_operation]
        weights = np.zeros(crt_model.flat_gradient_shape, float)
        req = comm.Irecv(weights, source=0, tag=0)
        req.Wait()
        weights = crt_model.unflatten(weights)
        crt_model.model.set_weights(weights)
        # print('Updated model parameters')
        # receive task details
        task_details = np.zeros_like([0, 0], dtype=int)
        req = comm.Irecv(task_details, source=0, tag=0)
        req.Wait()
        # print('task details', task_details)
        # receive tasks
        parts = [np.zeros(task_details[1], dtype=int) for _ in range(task_details[0])]
        coefficient = np.zeros(task_details[0], dtype=float)
        reqs = []
        for subpart in parts:
            reqs.append(comm.Irecv(subpart, source=0, tag=0))
        reqs.append(comm.Irecv(coefficient, source=0, tag=1))
        MPI.Request.waitall(reqs)
        # compute the gradient
        local_x_train = [x_train[subpart] for subpart in parts]
        local_y_train = [y_train[subpart] for subpart in parts]
        if straggling_status == 0:
            rep = 1
        else:
            rep = alpha
        init = time.time()
        for _ in range(rep):
            grad = crt_model.calculate_gradients(local_x_train, local_y_train, coefficient).numpy()
        time_spent_crt = time.time() - init
        # print('gradient computed')
        # print('grad at worker', grad)
        # tranmsit the result
        req = comm.Isend(np.ascontiguousarray(grad, dtype=float), dest=0, tag=0)
        req.Wait()
        comm.send(time_spent_crt, dest=0, tag=0)
        model_under_operation = (model_under_operation + 1) % len(models)





comm = MPI.COMM_WORLD
rank = comm.Get_rank()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 28, 28, 1)) / 255.
x_test = np.reshape(x_test, (-1, 28, 28, 1)) / 255.
batch_size_per_worker = 256
alpha = 5
tol = 0.9
num_slots = 6000
num_workers = 4
epsilon = 2
num_models = 4
lr_list = np.linspace(0.01, 0.1, num_models)
models = [Model(lr) for lr in lr_list]
a = 0.2
b = 0.8
num_states = 1
if rank == 0:
    master()
else:
    worker()
