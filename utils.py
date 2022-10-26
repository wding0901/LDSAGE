import numpy as np
import dgl


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 2048)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def rw_samling(rw_args, adj_mat, src_nodes, sample_nums):
    sampling_result = [src_nodes]
    num_traversals = rw_args[0]
    termination_prob = rw_args[1]
    num_random_walks = rw_args[2]
    G = dgl.from_scipy(adj_mat)

    for k, hopk_num in enumerate(sample_nums):
        sampler = dgl.sampling.RandomWalkNeighborSampler(G, num_traversals, termination_prob, num_random_walks, hopk_num)

        for i in range(3):
            neighbours, _ = sampler(sampling_result[k]).all_edges(form='uv')
            if neighbours.shape[0] == len(sampling_result[k]) * hopk_num:
                break
            print("sample fail")

        sampling_result.append(neighbours.numpy())
    return sampling_result
