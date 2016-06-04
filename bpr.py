import random
from itertools import chain, imap
from tqdm import tqdm
import numpy as np

def flatmap(f, items):
    return chain.from_iterable(imap(f, items))

def gradient_single_point(user_id, prod_id, prod_id_neg, 
                           user_mat, prod_mat, lambda_reg, alpha):
    x_uij = user_mat[user_id].dot(prod_mat[prod_id]) - \
            user_mat[user_id].dot(prod_mat[prod_id_neg])

    step_size = np.exp(-x_uij) / (1 + np.exp(-x_uij)) * alpha

    user_mat[user_id] += step_size * (prod_mat[prod_id] - prod_mat[prod_id_neg]) - lambda_reg * user_mat[user_id]
    prod_mat[prod_id] += step_size * user_mat[user_id] - lambda_reg * prod_mat[prod_id]
    prod_mat[prod_id_neg] -= step_size * user_mat[user_id] - lambda_reg * prod_mat[prod_id_neg]


def sample_optimize_partition(ratings, user_mat, prod_mat, num_prods, 
                                lambda_reg=0.001, alpha=0.1, position=None):

    NUM_OF_NEGATIVE_PER_IMPLICIT = 30

    positive_ratings_repeat = flatmap(lambda x: [x] * NUM_OF_NEGATIVE_PER_IMPLICIT, ratings)
    negative_ratings = map(lambda x: x[:2] + (np.random.randint(num_prods) + 1, ), positive_ratings_repeat)


    sampled_ratings = random.sample(negative_ratings, 20000)

    for u, i, j in tqdm(sampled_ratings, position=position, leave=False):
        gradient_single_point(u, i, j, user_mat, prod_mat, lambda_reg, alpha)
    yield user_mat, prod_mat

def optimize_mf(ratings, rank, num_iter=10):

    ratings_partitioned = ratings.partitionBy(4).persist()

    num_users = ratings_partitioned.map(lambda x: x[0]).max()
    num_songs = ratings_partitioned.map(lambda x: x[1]).max()

    user_mat = np.random.uniform(size=(num_users + 1, rank))
    prod_mat = np.random.uniform(size=(num_songs + 1, rank))

    for _ in xrange(num_iter):
        result = ratings_partitioned.mapPartitionsWithIndex(
            lambda ix, ratings: sample_optimize_partition(
                ratings, user_mat, prod_mat, num_songs, position=ix
            )
        ).persist()
        num = float(result.count())
        
        user_mat, prod_mat = result.reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))

        user_mat /= num
        prod_mat /= num

    return (user_mat, prod_mat)
