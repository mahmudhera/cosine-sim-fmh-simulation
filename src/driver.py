import numpy as np
from numpy import dot
from numpy.linalg import norm
import random
from tqdm import tqdm

def get_cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def get_sketch(vector, random_indices):
    return np.take(vector, random_indices)

def generate_random_indices(num_total_entries, num_random_indices):
    return random.sample(range(num_total_entries), num_random_indices)

def generate_random_vector(num_entries):
    #return np.random.randint(2, size=num_entries)
    num_ones = 10000
    num_zeros = 3000
    return np.random.choice( [0]*num_zeros+[1]*num_ones, num_entries )

def get_squared_length(u):
    return sum( [x**2 for x in u] )

if __name__ == '__main__':
    seed = 2
    num_entries = 50000
    fmh_sketch_size = 100
    num_simulations = 100000
    np.random.seed(seed)
    random.seed(seed)

    a = np.array([0,1,0])
    b = np.array([0,0,1])
    ans = 0
    assert ans == get_cosine_similarity(a,b)

    a = np.array([0,1,1])
    b = np.array([0,0,1])
    ans = 1.0/2**0.5
    assert ans == get_cosine_similarity(a,b)

    v = generate_random_vector(10)
    a = generate_random_indices(10, 4)
    b = generate_random_indices(10, 4)
    c = generate_random_indices(10, 4)

    print(v, a, get_sketch(v, a))
    print(v, b, get_sketch(v, b))
    print(v, c, get_sketch(v, c))

    print('------- STARTING SIMULATION ---------')

    for j in range(20):
        u = generate_random_vector(num_entries)
        v = generate_random_vector(num_entries)
        cosine_original_space = get_cosine_similarity(u, v)

        cosine_similarities = []
        for i in range(num_simulations):
            indices = generate_random_indices(num_entries, fmh_sketch_size)
            u_reduced = get_sketch(u, indices)
            v_reduced = get_sketch(v, indices)
            cosine_similarity_reduced_space = get_cosine_similarity(u_reduced, v_reduced)
            cosine_similarities.append(cosine_similarity_reduced_space)
        print( cosine_original_space, np.mean(cosine_similarities), np.var(cosine_similarities) )

    print('------- STARTING SIMULATION ---------')

    for j in range(20):
        u = generate_random_vector(num_entries)
        v = generate_random_vector(num_entries)

        u_reduced_list = []
        v_reduced_list = []
        for i in range(num_simulations):
            indices = generate_random_indices(num_entries, fmh_sketch_size)
            u_reduced = get_sketch(u, indices)
            v_reduced = get_sketch(v, indices)
            u_reduced_list.append(u_reduced)
            v_reduced_list.append(v_reduced)

        print( get_squared_length(u), np.mean( [get_squared_length(vector) for vector in u_reduced_list] ) )
        print( get_squared_length(v), np.mean( [get_squared_length(vector) for vector in v_reduced_list] ) )
