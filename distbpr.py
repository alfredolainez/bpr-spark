import random
import numpy as np


def _optimize_partition(user_ratings, prod_mat, nb_prods, l2_reg=0.001,
                        alpha=0.1, negative_samples=30, num_samples=20000):
    # yank everything out of the iterator
    user_ratings = [_ for _ in user_ratings]

    ratings = (
        (u_id, prod)
        for (u_id, (products, _)) in user_ratings
        for prod in products
    )

    user_vectors = {u: v for (u, (_, v)) in user_ratings}

    pos_repeated = (
        a
        for b in ([x] * negative_samples for x in ratings)
        for a in b
    )

    neg_ratings = [
        (x[0], x[1], np.random.randint(nb_prods) + 1)
        for x in pos_repeated
    ]

    shuff_ratings = neg_ratings
    random.shuffle(shuff_ratings)

    for u_id, pos_id, neg_id in shuff_ratings[:num_samples]:

        u_vector = user_vectors.get(u_id)
        x_uij = u_vector.dot(prod_mat[pos_id]) - u_vector.dot(prod_mat[neg_id])

        scale = np.exp(-x_uij) / (1 + np.exp(-x_uij))

        prod_mat[pos_id] += alpha * (
            (scale * u_vector) - l2_reg * prod_mat[pos_id])

        prod_mat[neg_id] += alpha * (
            (scale * u_vector) - l2_reg * prod_mat[neg_id])

        user_vectors[u_id] = u_vector + alpha * (
            scale * (prod_mat[pos_id] - prod_mat[neg_id]) - l2_reg * u_vector)

    yield (prod_mat, user_vectors.items())


def optimizeMF(ratings, rank=10, nb_iter=10, nb_partitions=4):
    """optimize BPR for Matrix Factorization
    
    Args:
    -----
        ratings: RDD of (user, item) implicit interactions
        rank: latent factor dimension
        nb_iter: how many iterations of SGD
        nb_partitions: how many user partitions to distribute
    
    Returns:
    --------
        (userMat, itemMat)
    """
    
    nb_prods = ratings.map(lambda (_, i): i).max()
    ratings_by_user = ratings.groupByKey().persist()

    def make_vec((u_id, products)):
        return (u_id, (products, np.random.uniform(size=rank)))

    user_ratings = ratings_by_user.map(make_vec).persist()
    ratings_partitioned = user_ratings.partitionBy(nb_partitions).persist()
    prod_mat = np.random.uniform(size=(nb_prods + 1, rank))

    for _ in xrange(nb_iter):
        result = ratings_partitioned.mapPartitions(
            lambda ratings: _optimize_partition(ratings, prod_mat, nb_prods)
        ).persist()

        prod_mat = result.map(
            lambda x: x[0]
        ).reduce(
            lambda x, y: x + y
        ) / result.count()

        user_vecs_rdd = result.map(lambda x: x[1]).flatMap(lambda x: x)
        ratings_partitioned = ratings_by_user.join(user_vecs_rdd)  # .persist()
        result.unpersist()

    # Only for evaluation purposes
    nb_users = ratings.map(lambda x: x[0]).max()

    user_mat = np.random.uniform(size=(nb_users + 1, rank))

    user_vectors = map(
        lambda (u_id, (products, vector)): (u_id, vector),
        ratings_partitioned.toLocalIterator()
    )

    for u, v in user_vectors:
        user_mat[u] = v

    return (user_mat, prod_mat)
