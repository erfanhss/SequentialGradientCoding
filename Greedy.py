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



class Job:
    def __init__(self, model_id, dataset_idx, round):
        self.model_id = model_id
        self.round = round
        self.pieces = self.divide_into_pieces(dataset_idx, num_workers)
        self.identifiers = np.arange(num_workers)
        self.piece_map = [[(worker_idx + i) % num_workers for i in range(epsilon + 1)] for worker_idx in
                          range(num_workers)]
        complement_piece_map = [[(worker_idx + 1 + i) % num_workers for i in range(num_workers - epsilon - 1)] for
                                worker_idx
                                in range(num_workers)]
        self.coefficients = [
            [np.prod([self.identifiers[worker_idx] - self.identifiers[j] for j in complement_piece_map[idx]]) for idx in
             self.piece_map[worker_idx]] for worker_idx in range(num_workers)]

        self.failure_map = np.zeros(num_workers)
        self.coded_results = []


    def set_failure_status(self, worker_idx):
        self.failure_map[worker_idx] = 1

    def divide_into_pieces(self, array, num_parts):
        tmp = np.copy(array)
        if len(array) % num_parts != 0:
            missing = num_parts - (len(array) % num_parts)
            to_be_added_idx = np.random.permutation(len(array))[0:missing]
            tmp = np.concatenate((array, array[to_be_added_idx]))
        return np.array_split(tmp, num_parts)

    def get_minitask(self, worker_idx):
        to_go_idx = [self.pieces[i] for i in self.piece_map[worker_idx]]
        coefficients_to_go = self.coefficients[worker_idx]
        return [[to_go_idx, coefficients_to_go], self.model_id]

    def push_result(self, result, worker_idx):
        self.coded_results.append([result, worker_idx])


    def calculate_result(self):
        if len(self.coded_results) < num_workers - s:
            print('ERROR: not enough coded pieces received. Got: ', len(self.coded_results), 'Need: ', num_workers - s)
        pieces = [self.coded_results[i][0] for i in range(num_workers-epsilon)]
        idx = [self.coded_results[i][1] for i in range(num_workers-epsilon)]
        rec_ident = [self.identifiers[i] for i in idx]
        vandermonde_matrix = np.array([[j ** i for j in rec_ident] for i in range(num_workers - epsilon)])
        inv_vand_mat = np.linalg.inv(vandermonde_matrix)
        recon_coef = inv_vand_mat[:, -1]
        recon = np.sum([recon_coef[i] * pieces[i] for i in range(num_workers - epsilon)], axis=0)
        return recon

def check_window(window):
    straggling_workers = 0
    for worker_idx in range(num_workers):
        location_ones = np.where(window[worker_idx, :])[0]
        if len(location_ones) == 0:
            continue
        burst_length = location_ones[-1] - location_ones[0] + 1
        straggling_workers += 1
        if burst_length > B:
            return 0
    if straggling_workers > epsilon:
        return 0
    return 1


def master():

    print('System specifications:')
    print('n', num_workers)
    print('B', B)
    print('W', W)
    print('epsilon', epsilon)
    print('x', x)
    job_queue = []
    model_under_operation = 0
    straggling_map = np.zeros([num_workers, num_slots * 2])
    time_spent = np.zeros_like(straggling_map)
    round_times = np.zeros(num_slots)
    for slot in range(num_slots):
        print(slot)
        # add new job to the queue
        idx = np.random.permutation(len(x_train))[0:num_workers * batch_size_per_worker]
        job_queue.append(Job(model_under_operation, idx, slot))
        # transmit minitasks
        minitasks = []
        # update parameters in the workers
        crt_model = models[model_under_operation]
        params = crt_model.flatten_gradients(crt_model.model.get_weights())
        param_req = []
        for worker_idx in range(num_workers):
            param_req.append(comm.Isend(np.ascontiguousarray(params, dtype=float), dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(param_req)
        # print('Parameters updated')
        # create minitask
        left_over = 0
        job_pieces_received = []
        job_idx_per_worker = np.zeros(num_workers, dtype=int)
        for job in job_queue:
            job_pieces_received.append(len(job.coded_results))
        for worker_idx in range(num_workers):
            reattempt = 0
            for job_idx, prev_job in enumerate(job_queue):
                if (slot - prev_job.round)%B == 0 and prev_job.round < slot:
                    if job_pieces_received[job_idx] < num_workers - s:
                        minitasks.append(prev_job.get_minitask(worker_idx))
                        job_pieces_received[job_idx] += 1
                        reattempt = 1
                        left_over = 1
                        job_idx_per_worker[worker_idx] = int(job_idx)
                        break
            if reattempt == 0:
                minitasks.append(job_queue[-1].get_minitask(worker_idx))
                job_idx_per_worker[worker_idx] = int(len(job_queue) - 1)
        # print(minitasks)
        print(job_idx_per_worker)
        # print('Minitasks created')
        # transmit minitasks specifications
        reqs = []
        for worker_idx in range(num_workers):
            minitask = minitasks[worker_idx]
            reqs.append(comm.Isend(np.array([len(minitask[0][0]), len(minitask[0][0][0]), minitask[1]]),
                                   dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(reqs)
        # print('Minitasks details transmitted')
        # transmit content
        reqs = []
        for worker_idx in range(num_workers):
            minitask = minitasks[worker_idx]
            for part in minitask[0][0]:
                reqs.append(comm.Isend(np.ascontiguousarray(part, dtype=int), dest=worker_idx + 1, tag=0))
            reqs.append(comm.Isend(np.ascontiguousarray(minitask[0][1], dtype=float), dest=worker_idx + 1, tag=1))
        MPI.Request.waitall(reqs)
        # print('Minitasks transmitted')
        # get back the results
        reqs = []
        results = [np.zeros(crt_model.flat_gradient_shape) for _
                   in range(num_workers)]
        for worker_idx in range(num_workers):
            reqs.append(comm.Irecv(results[worker_idx], source=worker_idx+1, tag=0))
        MPI.Request.waitall(reqs)
        for worker_idx in range(num_workers):
            time_spent[worker_idx, slot] = comm.recv(source=worker_idx+1, tag=0)
        # print('Results received')
        # determine stragglers
        crt_round_times = time_spent[:, slot]
        sorted_idx_round_times = np.argsort(crt_round_times)

        for idx in sorted_idx_round_times:
            if crt_round_times[idx] > (1+tol)*crt_round_times[sorted_idx_round_times[0]]:
                straggling_map[idx, slot] = 1
                if check_window(straggling_map[:, max(0, slot-(W)+1):slot+1]) != 1:
                    straggling_map[:, slot] = 0
                    break
        print(straggling_map[:, max(0, slot-(W)+1):slot+1])
        for worker_idx in range(num_workers):
            if straggling_map[worker_idx, slot] == 0:
                # print(job_idx_per_worker[worker_idx])
                # print(slot)
                # print(worker_idx)
                job_queue[job_idx_per_worker[worker_idx]].push_result(
                    results[worker_idx], worker_idx
                )
        # finalizing the round
        model_under_operation += 1
        model_under_operation = model_under_operation % len(models)
        if slot - job_queue[0].round == x*B:
            print('Minitask completed')
            out = job_queue.pop(0)
            result = out.calculate_result()
            models[out.model_id].update_params(result)
            print('Parameters of model updated', out.model_id)
    straggling_map = straggling_map[:, 0:num_slots]

    time_spent = time_spent[:, 0:num_slots]
    wait_map = 1 - straggling_map
    effective_time = np.multiply(time_spent, wait_map)
    round_time = np.max(effective_time, axis=0)
    print(np.mean(round_time))
    np.save('straggling_map_greedy', straggling_map)
    np.save('time_spent_greedy', time_spent)
    for idx, model in enumerate(models):
        print(model.report_performance())
        np.save('Model_'+str(idx)+'greedy_test_loss', model.loss)
        np.save('Model_' + str(idx) + 'greedy_test_accuracy', model.accuracy)

def worker():
    model_under_operation = 0
    state = 0
    seed_arr = np.random.RandomState(seed=rank).randint(0, 100000, size=num_slots)
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
                state = (state + 1) % (num_states+1)
        # receive new parameters from master
        crt_model = models[model_under_operation]
        weights = np.zeros(crt_model.flat_gradient_shape, float)
        req = comm.Irecv(weights, source=0, tag=0)
        req.Wait()
        weights = crt_model.unflatten(weights)
        crt_model.model.set_weights(weights)
        model_under_operation += 1
        model_under_operation = model_under_operation % len(models)
        # receive minitaks details
        minitaks_details = np.empty_like([0, 0, 0])
        req = comm.Irecv(minitaks_details, source=0, tag=0)
        req.Wait()
        # print('worker', rank, 'has minitasks details', minitaks_details_arr)
        req = []
        minitasks = [np.zeros(minitaks_details[1], dtype=int) for _ in range(minitaks_details[0])]
        coefficients = np.zeros(minitaks_details[0], dtype=float)
        for buffer in minitasks:
            req.append(comm.Irecv(buffer, source=0, tag=0))
        req.append(comm.Irecv(coefficients, source=0, tag=1))
        # print('worker', len(req))
        MPI.Request.waitall(req)
        # print('got all poxs')
        init = time.time()
        results = []
        if straggling_status == 0:
            rep = 1
        else:
            rep = alpha
        x_train_crt = [x_train[idx] for idx in minitasks]
        y_train_crt = [y_train[idx] for idx in minitasks]
        for _ in range(rep):
            tmp = models[minitaks_details[2]].calculate_gradients(x_train_crt, y_train_crt, coefficients).numpy()
        results.append(tmp)
        time_spent = time.time() - init
        # transmit results back to the master
        req = []
        for tag, res in enumerate(results):
            req.append(comm.Isend(np.ascontiguousarray(res, dtype=float), dest=0, tag=tag))
        MPI.Request.waitall(req)
        comm.send(time_spent, dest=0, tag=0)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 28, 28, 1)) / 255.
x_test = np.reshape(x_test, (-1, 28, 28, 1)) / 255.
batch_size_per_worker = 256
alpha = 5
tol = 0.9
num_slots = 5000
num_workers = 4
x = 4
epsilon = 2
B = 1
W = x*B+1
s = np.ceil((B*epsilon)/(W-1+B))
num_models = 8
lr_list = np.linspace(0.01, 0.1, num_models)
models = [Model(lr) for lr in lr_list]
a = 0.05
b = 0.8
num_states = 1
if rank == 0:
    master()
else:
    worker()
