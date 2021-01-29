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
    def __init__(self, model_id, dataset_idx):
        self.model_id = model_id
        self.num_data_points = len(dataset_idx)
        p1_len = int(len(dataset_idx) * x)
        dataset_idx1 = dataset_idx[0:p1_len]
        dataset_idx2 = dataset_idx[p1_len:]
        num_subparts_p1 = num_workers * (W - 1)
        self.part1_pieces = self.divide_into_pieces(dataset_idx1, num_subparts_p1)
        self.round = 0
        round_splits_part2 = self.divide_into_pieces(dataset_idx2, B)
        self.part2_pieces = [self.divide_into_pieces(split, num_workers) for split in round_splits_part2]
        self.identifiers = np.arange(num_workers)
        self.piece_map = [[(worker_idx + i) % num_workers for i in range(epsilon + 1)] for worker_idx in
                          range(num_workers)]
        complement_piece_map = [[(worker_idx + 1 + i) % num_workers for i in range(num_workers - epsilon - 1)] for
                                worker_idx
                                in range(num_workers)]
        self.coefficients = [
            [np.prod([self.identifiers[worker_idx] - self.identifiers[j] for j in complement_piece_map[idx]]) for idx in
             self.piece_map[worker_idx]] for worker_idx in range(num_workers)]
        self.failure_map = np.zeros([num_workers, W + B - 1])
        self.uncoded_results = []
        self.coded_results = [[] for _ in range(B)]
        self.content_map = np.zeros([num_workers, W+B-1], dtype=int)
    def next_round(self):
        # record success of reattempted pieces
        if self.round >= W-1:
            for worker_idx in range(num_workers):
                if self.failure_map[worker_idx, self.round] == 0:
                    if self.content_map[worker_idx, self.round] >= 0:
                        # print('pox', self.content_map[worker_idx, self.round])
                        self.failure_map[worker_idx, self.content_map[worker_idx, self.round]] = 0
        self.round += 1


    def set_failure_status(self, round, worker_idx):
        self.failure_map[worker_idx, round] = 1

    def divide_into_pieces(self, array, num_parts):
        tmp = np.copy(array)
        if len(array) % num_parts != 0:
            missing = num_parts - (len(array) % num_parts)
            to_be_added_idx = np.random.permutation(len(array))[0:missing]
            tmp = np.concatenate((array, array[to_be_added_idx]))
        return np.array_split(tmp, num_parts)

    def get_minitask(self, worker_idx):
        if self.round < W - 1:
            # uncoded tasks
            result = self.part1_pieces[self.round * num_workers + worker_idx]
            self.content_map[worker_idx, self.round] = self.round
            return ['uncoded', result, self.model_id, None]
        else:
            # coded parts and reattempts
            part2_round_num = self.round - (W-1)
            if self.failure_map[worker_idx, part2_round_num] == 1:
                result = self.part1_pieces[part2_round_num * num_workers + worker_idx]
                self.content_map[worker_idx, self.round] = part2_round_num
                # print('reattempting due to prior failure', 'current round', self.round, 'attempted idx', part2_round_num)
                return ['uncoded', result, self.model_id, None]
            if np.sum(self.failure_map[worker_idx, B:W-1]) > 0:
                focused_slice = self.failure_map[worker_idx, B:W-1]
                to_go = np.where(focused_slice == 1)[0][0] + B
                self.content_map[worker_idx, self.round] = to_go
                result = self.part1_pieces[to_go * num_workers + worker_idx]
                # print('reattempting due to prior failure', 'current round', self.round, 'attempted idx', to_go)
                return ['uncoded', result, self.model_id, None]

            # focused_slice = self.failure_map[worker_idx, 0:W-1]
            # failure_idx = np.where(focused_slice == 1)[0]

            # if len(failure_idx) != 0:
            #     idx_to_reattempt = np.where(failure_idx % B == part2_round_num)[0]
            #     if len(idx_to_reattempt) != 0:
            #         idx_to_reattempt = failure_idx[idx_to_reattempt[0]]
            #         # print('reattempting due to prior failure', 'current round', self.round, 'attempted idx',
            #         #       idx_to_reattempt)

            pieces = self.part2_pieces[part2_round_num]
            to_go_idx = [pieces[i] for i in self.piece_map[worker_idx]]
            coefficients_to_go = self.coefficients[worker_idx]
            self.content_map[worker_idx, self.round] = -1
            return ['coded', [to_go_idx, coefficients_to_go], self.model_id, part2_round_num]

    def push_result(self, result, type, worker_idx, round_idx):
        if type == 'uncoded':
            self.uncoded_results.append(result)
        else:
            self.coded_results[round_idx].append([result, worker_idx])
    def calculate_result(self):
        if len(self.uncoded_results) != num_workers*(W-1):
            print('ERROR: not enough uncoded pieces received. Got: ', len(self.uncoded_results), 'need: ', num_workers*(W-1))
            exit(0)
        coded_results = []
        for round_idx, round_pieces in enumerate(self.coded_results):
            if len(round_pieces) < num_workers - epsilon:
                print('ERROR: not enough coded pieces received. Got: ', len(round_pieces), 'Need: ', num_workers - epsilon)
            pieces = [round_pieces[i][0] for i in range(num_workers-epsilon)]
            idx = [round_pieces[i][1] for i in range(num_workers-epsilon)]
            rec_ident = [self.identifiers[i] for i in idx]
            vandermonde_matrix = np.array([[j ** i for j in rec_ident] for i in range(num_workers - epsilon)])
            inv_vand_mat = np.linalg.inv(vandermonde_matrix)
            recon_coef = inv_vand_mat[:, -1]
            recon = np.sum([recon_coef[i] * pieces[i] for i in range(num_workers - epsilon)], axis=0)
            # print(recon)
            # print(models[self.model_id].calculate_gradients([x_train[np.concatenate(self.part2_pieces[round_idx])]],
            #                                                 [y_train[np.concatenate(self.part2_pieces[round_idx])]],
            #                                                 coefficients=[1]))
            coded_results.append(recon)
        uncoded_res = np.sum(self.uncoded_results, axis=0)
        # print(uncoded_res)
        # print(models[self.model_id].calculate_gradients([x_train[np.concatenate(self.part1_pieces)]],
        #                                                 [y_train[np.concatenate(self.part1_pieces)]],
        #                                                 coefficients=[1]))

        coded_res = np.sum(coded_results, axis=0)
        result = (uncoded_res + coded_res) / self.num_data_points
        return result

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
    for slot in range(num_slots):
        print(slot)
        # add new job to the queue
        idx = np.random.permutation(len(x_train))[0:num_workers * batch_size_per_worker]
        job_queue.append(Job(model_under_operation, idx))
        if job_queue[0].round == W - 1 + B:
            out = job_queue.pop(0)
            result = out.calculate_result()
            models[out.model_id].update_params(result)
        # transmit minitasks
        minitasks = [[] for _ in range(num_workers)]
        # update parameters in the workers
        crt_model = models[model_under_operation]
        params = crt_model.flatten_gradients(crt_model.model.get_weights())
        param_req = []
        for worker_idx in range(num_workers):
            param_req.append(comm.Isend(np.ascontiguousarray(params, dtype=float), dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(param_req)
        # create minitask
        for job in job_queue:
            for worker_idx in range(num_workers):
                minitasks[worker_idx].append(job.get_minitask(worker_idx))
        # transmit number of minitasks
        reqs = []
        for worker_idx in range(num_workers):
            comm.Isend(np.array([len(minitasks[worker_idx])]), dest=worker_idx + 1, tag=0)
        MPI.Request.waitall(reqs)
        # transmit minitasks specifications
        reqs = []
        for worker_idx in range(num_workers):
            for minitask in minitasks[worker_idx]:
                if minitask[0] == 'uncoded':
                    reqs.append(comm.Isend(np.array([1, len(minitask[1]), minitask[2]]), dest=worker_idx + 1, tag=0))
                else:
                    reqs.append(comm.Isend(np.array([len(minitask[1][0]), len(minitask[1][0][0]), minitask[2]]),
                                           dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(reqs)
        # transmit content
        reqs = []
        for worker_idx in range(num_workers):
            for minitask in minitasks[worker_idx]:
                if minitask[0] == 'uncoded':
                    # datapoint idx for gradient computation
                    # print('being transmitted to', worker_idx+1, np.ascontiguousarray(minitask[1]))
                    reqs.append(comm.Isend(np.ascontiguousarray(minitask[1]), dest=worker_idx + 1, tag=0))
                    # coefficients for gradient scaling
                    reqs.append(comm.Isend(np.array([1.]), dest=worker_idx + 1, tag=1))
                else:
                    for part in minitask[1][0]:
                        reqs.append(comm.Isend(np.ascontiguousarray(part, dtype=int), dest=worker_idx + 1, tag=0))
                    reqs.append(comm.Isend(np.ascontiguousarray(minitask[1][1], dtype=float), dest=worker_idx + 1, tag=1))
        # print(len(reqs))
        MPI.Request.waitall(reqs)

        # get back the results
        reqs = []
        results = [[np.zeros(crt_model.flat_gradient_shape) for _ in range(len(minitasks[worker_idx]))] for worker_idx
                   in range(num_workers)]
        for worker_idx in range(num_workers):
            for tag, res in enumerate(results[worker_idx]):
               reqs.append(comm.Irecv(res, source=worker_idx+1, tag=tag))
        MPI.Request.waitall(reqs)
        for worker_idx in range(num_workers):
            time_spent[worker_idx, slot] = comm.recv(source=worker_idx+1, tag=0)
        crt_round_times = time_spent[:, slot]
        sorted_idx_round_times = np.argsort(crt_round_times)

        for idx in sorted_idx_round_times:
            if crt_round_times[idx] > (1+tol)*crt_round_times[sorted_idx_round_times[0]]:
                straggling_map[idx, slot] = 1
                # print('pox', straggling_map[:, max(0, slot-(W-1+B)+1):slot+1])
                # print(check_window(straggling_map[:, max(0, slot-(W-1+B)+1):slot+1]))
                if check_window(straggling_map[:, max(0, slot-(W)+1):slot+1]) != 1:
                    straggling_map[:, slot] = 0
                    break
        # print(straggling_map[:, max(0, slot - (W+B-1) + 1):slot + 1])
        for idx in sorted_idx_round_times:
            if straggling_map[idx, slot] == 1:
                for job in job_queue:
                    job.failure_map[idx, job.round] = 1
        # print(time_spent[:, 0:slot + 1])
        # print(straggling_map[:, 0:slot + 1])
        for worker_idx in range(num_workers):
            for idx, res in enumerate(results[worker_idx]):
                if straggling_map[worker_idx, slot] == 0:
                    # print(idx, worker_idx)
                    job_queue[idx].push_result(res, minitasks[worker_idx][idx][0], worker_idx, minitasks[worker_idx][idx][3])
        # finalizing the round
        model_under_operation += 1
        model_under_operation = model_under_operation % (W + B - 1)
        for job in job_queue:
            job.next_round()
    straggling_map = straggling_map[:, 0:num_slots]

    time_spent = time_spent[:, 0:num_slots]
    wait_map = 1 - straggling_map
    effective_time = np.multiply(time_spent, wait_map)
    round_time = np.max(effective_time, axis=0)
    print(np.mean(round_time))
    np.save('straggling_map_seq', straggling_map)
    np.save('time_spent_seq', time_spent)
    for idx, model in enumerate(models):
        print(model.report_performance())
        np.save('Model_'+str(idx)+'sequential_test_loss', model.accuracy)
        np.save('Model_' + str(idx) + 'sequential_test_accuracy', model.accuracy)

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
        model_under_operation = model_under_operation % (W + B - 1)
        # print('Model parameter updated in worker', rank)
        # receive number of minitaks
        minitaks_len = np.empty_like([0])
        req = comm.Irecv(minitaks_len, source=0, tag=0)
        req.Wait()
        minitaks_len = minitaks_len[0]
        # print('Number of minitasks worker', rank, 'is', minitaks_len)
        # receive minitaks details
        req = []
        minitaks_details_arr = [np.empty_like([0, 0, 0]) for _ in range(minitaks_len)]
        model_ids = []
        for minitaks_details in minitaks_details_arr:
            req.append(comm.Irecv(minitaks_details, source=0, tag=0))
        MPI.Request.waitall(req)
        for minitaks_details in minitaks_details_arr:
            model_ids.append(minitaks_details[2])
        # print('worker', rank, 'has minitasks details', minitaks_details_arr)
        req = []
        minitasks = [[np.zeros(minitaks_details_arr[minitask_idx][1], dtype=int) for _ in
                      range(minitaks_details_arr[minitask_idx][0])] for minitask_idx in range(minitaks_len)]
        coefficients = [np.zeros(minitaks_details_arr[minitask_idx][0], dtype=float) for minitask_idx in range(minitaks_len)]
        for minitask in minitasks:
            for buffer in minitask:
                req.append(comm.Irecv(buffer, source=0, tag=0))
        for coefficient in coefficients:
            req.append(comm.Irecv(coefficient, source=0, tag=1))
        # print('worker', len(req))
        MPI.Request.waitall(req)
        # print('got all poxs')
        init = time.time()
        results = []
        if straggling_status == 0:
            rep = 1
        else:
            rep = alpha
        for (minitask, coefficient, id) in zip(minitasks, coefficients, model_ids):
            x_train_crt = [x_train[idx] for idx in minitask]
            y_train_crt = [y_train[idx] for idx in minitask]
            for _ in range(rep):
                tmp = models[id].calculate_gradients(x_train_crt, y_train_crt, coefficient).numpy()
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
alpha = 10
tol = 0.9
num_slots = 5000
num_workers = 4
W = 7
epsilon = 3
B = 2
num_models = W+B-1
x = (epsilon + 1) * (W - 1) / (B + W - 1 + epsilon * (W - 1))
lr_list = np.linspace(0.01, 0.1, num_models)
models = [Model(lr) for lr in lr_list]
a = 0.05
b = 0.8
num_states = 1
if rank == 0:
    master()
else:
    worker()
