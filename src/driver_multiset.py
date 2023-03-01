import numpy as np
from numpy import dot
from numpy.linalg import norm
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd

def get_cosine_similarity(a, b):
    try:
        return dot(a, b)/(norm(a)*norm(b))
    except:
        return 0

def get_sketch(vector, random_indices):
    return np.take(vector, random_indices)

def generate_random_indices(num_total_entries, num_random_indices):
    return random.sample(range(num_total_entries), num_random_indices)

def generate_random_binary_vector(num_entries, num_ones):
    ones = [1] * num_ones
    zeros = [0] * (num_entries - num_ones)
    all = ones + zeros
    random.shuffle(all)
    return all

def generate_random_multi_vector(num_entries, max_value, num_zeros=0):
    # will generate a random vector with values (0,1,2,...,max_value-1)
    # these values all are equally likely if distr
    zeros = [0] * num_zeros
    num_non_zero_entries = max(num_entries-num_zeros, 0)
    non_zeros = random.sample(list(range(1, max_value+1))*(int(num_non_zero_entries/max_value)+2), num_non_zero_entries)
    all = zeros + non_zeros
    random.shuffle(all)
    return all

def get_squared_length(u):
    return sum( [x**2 for x in u] )

if __name__ == '__main__':
    seed = 0
    num_entries = 10000
    num_simulations = 10000
    np.random.seed(seed)
    random.seed(seed)

    max_value_u = 100
    max_value_v_low = 50
    max_value_v_high = 150

    scale_factors = [0.0001, 0.001, 0.01, 0.1]

    print('------- STARTING SIMULATION ---------')

    lst = []

    for scale_factor in scale_factors:
        # simulate hash function, get the indices
        sketch_size = int(scale_factor * num_entries)
        indices = generate_random_indices (num_entries, sketch_size)

        print(f"Running for scale factor: {scale_factor}")

        for i in tqdm(range(num_simulations)):
            # generate u and v randomly
            u = generate_random_multi_vector(num_entries, max_value_u)
            max_value_v = np.random.randint(max_value_v_low, max_value_v_high+1)
            num_zeros_v = np.random.randint(0, num_entries+1)
            v = generate_random_multi_vector(num_entries, max_value_v, num_zeros=num_zeros_v)

            # get original cosine similarity
            cosine_original_space = get_cosine_similarity(u, v)

            # get u' and v', and compute their similarity
            u_reduced = get_sketch(u, indices)
            v_reduced = get_sketch(v, indices)
            cosine_similarity_reduced_space = get_cosine_similarity(u_reduced, v_reduced)

            # record original and estimated
            lst.append( (scale_factor, max_value_u, max_value_v, cosine_original_space, cosine_similarity_reduced_space) )

    df = pd.DataFrame(lst, columns=['s', 'max_val_u', 'max_val_v', 'cos_org', 'cos_fmh'])
    df.to_csv('cosine_simulation_results_multiset')
