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

    def calculate_gradients(self, x_train, y_train):
        with tf.GradientTape() as tape:
            logits = self.model(x_train, training=True)
            loss_value = self.loss_fn(y_train, logits)
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

def check_window(window, T, s):
    if np.sum(window) > (T+1)*s:
        return 0
    return 1


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
    round_times = []
    straggling_map = np.zeros([num_workers, num_slots])
    wait_map = np.zeros_like(straggling_map)
    gradient_queue = [[np.zeros(models[0].flat_gradient_shape) for _ in range(num_workers)] for _ in range(T+1)]
    receive_queue = [[np.zeros(models[0].flat_gradient_shape) for _ in range(num_workers)] for _ in range(T+1)]

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
        piece_map = [[(worker_idx + i) % num_workers for i in range(s + 1)] for worker_idx in
                     range(num_workers)]
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
        MPI.Request.waitall(reqs)
        #### TODO: Rest of processing
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
        straggling_map[:, slot] = np.random.binomial(1, p, num_workers)
        wait_map[:, slot] = straggling_map[:, slot]
        if check_window(wait_map[:, max(0, slot-T):slot+1]) == 0:
            wait_map[:, slot] = 0
        receive_queue.pop(0)
        receive_queue.append(results)
        # compute the update
        if slot >= T:
            l_tild = [[np.zeros(crt_model.flat_gradient_shape) for _ in range(num_workers)] for _ in range(T+1)]
            for idx, l in enumerate(receive_queue):
                if idx == 0:
                    sum0 = np.sum(gradient_queue[0], axis=0)
                    sum1 = np.sum(gradient_queue[1], axis=0)
                    l_tild[idx] = [l[j] - sum0*y2[j]-sum1*y1[j] for j in range(num_workers)]
                if idx == 1:
                    sum1 = np.sum(gradient_queue[1], axis=0)
                    l_tild[idx] = [l[j] - sum1 * y2[j] for j in range(num_workers)]
                else:
                    l_tild[idx] = l

            if np.sum(wait_map[:, slot-2]) < 3:
                chosen_worker_idx = []
                for worker_idx in range(num_workers):
                    if len(chosen_worker_idx) < 2:
                        if wait_map[worker_idx, slot-2] == 0:
                            chosen_worker_idx.append(worker_idx)
                    else:
                        break
                tmp = S[:, [8+chosen_worker_idx[0], 8+chosen_worker_idx[1]]]
                recon_coefss = np.linalg.inv(tmp[[0, 5], :])[:, -1]
                recon = np.zeros(crt_model.flat_gradient_shape)
                for j in range(2):
                    recon += l_tild[0][chosen_worker_idx[j]]*recon_coefss[j]
            elif np.sum(wait_map[:, slot-1] < 2):
                chosen_worker_idx = []
                for worker_idx in range(num_workers):
                    if len(chosen_worker_idx) < 3:
                        if wait_map[worker_idx, slot - 2] == 0:
                            chosen_worker_idx.append(worker_idx)
                    else:
                        break
                tmp = S[:, [4 + chosen_worker_idx[0], 4 + chosen_worker_idx[1], 4 + chosen_worker_idx[2]]]
                recon_coefss = np.linalg.inv(tmp[[0, 4, 5], :])[:, -1]
                recon = np.zeros(crt_model.flat_gradient_shape)
                for j in range(3):
                    recon += l_tild[1][chosen_worker_idx[j]] * recon_coefss[j]
            else:
                chosen_worker_idx = []
                chosen_columns = []
                for t in range(T+1):
                    for worker_idx in range(num_workers):
                        if len(chosen_worker_idx) < 6:
                            if wait_map[worker_idx, t] == 0:
                                chosen_worker_idx.append([worker_idx, t])
                                if t == 0:
                                    chosen_columns.append(worker_idx+8)
                                elif t==1:
                                    chosen_columns.append(worker_idx+4)
                                else:
                                    chosen_columns.append(worker_idx)
                        else:
                            break

                recon_coefss = np.linalg.inv(
                    S[:, chosen_columns])[:, -1]
                recon = np.zeros(crt_model.flat_gradient_shape)
                for j in range(6):
                    recon += l_tild[chosen_worker_idx[j][1]][chosen_worker_idx[j][0]] * recon_coefss[j]
        crt_model.update_params(recon/num_workers/batch_size_per_worker)
        model_under_operation = (model_under_operation + 1) % len(models)
        gradient_queue.pop(0)
        gradient_queue.append(recon)
    for idx, model in enumerate(models):
        print(model.report_performance())
        np.save('Model_' + str(idx) + 'grad_code_test_loss', model.loss)
        np.save('Model_' + str(idx) + 'grad_code_test_accuracy', model.accuracy)
    np.save('round_times_grad_coding', np.array(round_times))
    # print(round_times)


def worker():
    model_under_operation = 0
    gradient_buffer = [[np.zeros(models[0].flat_gradient_shape) for _ in range(s+1)] for _ in range(T+1)]
    coeffs = []
    piece_map = [(rank - 1 + i) % num_workers for i in range(s + 1)]
    row = G2[:, rank-1]
    tmp_coeffs = np.array([row[x] for x in piece_map])
    coeffs.append(tmp_coeffs)
    row = G1[:, rank - 1]
    tmp_coeffs = np.array([row[x] for x in piece_map])
    coeffs.append(tmp_coeffs)
    row = G0[:, rank - 1]
    tmp_coeffs = np.array([row[x] for x in piece_map])
    coeffs.append(tmp_coeffs)

    for slot in range(num_slots):
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
        reqs = []
        for subpart in parts:
            reqs.append(comm.Irecv(subpart, source=0, tag=0))
        MPI.Request.waitall(reqs)
        # compute the gradient
        gradient_buffer.pop(0)
        gradient_buffer.append([np.zeros(crt_model.flat_gradient_shape) for _ in range(s+1)])
        init = time.time()
        for part_idx, part in parts:
            grad = crt_model.calculate_gradients(x_train[part], y_train[part]).numpy()
            gradient_buffer[-1][part_idx] = grad
        time_spent_crt = time.time() - init
        # print('gradient computed')
        # print('grad at worker', grad)
        # tranmsit the result
        result = np.zeros(crt_model.flat_gradient_shape)
        for t in range(T+1):
            for idx in range(s+1):
                result += gradient_buffer[t][idx] * coeffs[t][idx]
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
num_slots = 6000
num_workers = 4
num_models = 3
T = 2
s = 2
lr_list = np.linspace(0.01, 0.1, num_models)
models = [Model(lr) for lr in lr_list]
p = 0.3

### Generating encoding/decoding matrices
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
C = np.zeros([4, 4])
for i in range(4):
    for j in range(4):
        C[i, j] = 1 / (x[i] - y[j])
# Constructing S########
X = C[0, :]
y0 = C[1, :]
y1 = C[2, :]
y2 = C[3, :]
# print(C)
S = np.zeros([6, 12])
S[2, 0:4] = X
S[1, 4:8] = X
S[0, 8:12] = X
S[3, 0:4] = y0
S[4, 0:4] = y1
S[5, 0:4] = y2
S[4, 4:8] = y0
S[5, 4:8] = y1
S[5, 8:12] = y0
###Construction of G0#####
Delta0 = np.zeros([2, 4])
Delta0[0, :] = X
Delta0[1, :] = y0
Lambda0 = np.zeros([4, 2])
Lambda0[:, 1] = 1
for i in range(4):
    indx = (i + 1) % 4
    Lambda0[i, 0] = -Delta0[1, indx] / Delta0[0, indx]
G0 = np.dot(Lambda0, Delta0)
print(G0)
###Construction of G1#####
Delta1 = np.zeros([2, 4])
Delta1[0, :] = X
Delta1[1, :] = y1
Lambda1 = np.zeros([4, 2])
Lambda1[:, 1] = 1
for i in range(4):
    indx = (i + 1) % 4
    Lambda1[i, 0] = -Delta1[1, indx] / Delta1[0, indx]
G1 = np.dot(Lambda1, Delta1)
print(G1)
###Construction of G2#####
Delta2 = np.zeros([2, 4])
Delta2[0, :] = X
Delta2[1, :] = y2
Lambda2 = np.zeros([4, 2])
Lambda2[:, 1] = 1
for i in range(4):
    indx = (i + 1) % 4
    Lambda2[i, 0] = -Delta2[1, indx] / Delta2[0, indx]
G2 = np.dot(Lambda2, Delta2)
print(G2)


if rank == 0:
    master()
else:
    worker()
