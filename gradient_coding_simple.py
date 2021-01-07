import numpy as np

num_workers = 8
s = 6
identifiers = np.arange(0, num_workers)
dataset = np.random.randint(0, 1000, size=num_workers*5)
dataset_pieces = np.array_split(dataset, num_workers)
piece_map = [[(worker_idx + i) % num_workers for i in range(s + 1)] for worker_idx in range(num_workers)]
complement_piece_map = [[(worker_idx+1+i)%num_workers for i in range(num_workers-s-1)] for worker_idx in range(num_workers)]
coefficients = [[np.prod([identifiers[worker_idx] - identifiers[j] for j in complement_piece_map[idx]]) for idx in
                 piece_map[worker_idx]] for worker_idx in range(num_workers)]
result = [np.sum([coefficients[worker_idx][i] * dataset_pieces[piece_map[worker_idx][i]] for i in range(s + 1)], axis=0)
          for worker_idx in range(num_workers)]

received = [0, 2]
rec_res = [result[i] for i in received]
rec_ident = [identifiers[i] for i in received]
vandermonde_matrix = np.array([[j**i for j in rec_ident] for i in range(num_workers-s)])
inv_vand_mat = np.linalg.inv(vandermonde_matrix)
recon_coef = inv_vand_mat[:, -1]
recon = np.sum([recon_coef[i]*rec_res[i] for i in range(num_workers-s)], axis=0)
print(np.sum(dataset_pieces, axis=0))
print(recon)