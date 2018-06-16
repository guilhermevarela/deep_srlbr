import tensorflow as tf


def multiclass_loss(scores, pred, y):
    t_indices = get_indices(scores)
    t_scores_p = gather(scores, t_indices, pred)
    t_scores_y = gather(scores, t_indices, y)
    t_cost = tf.reduce_sum(t_scores_p) - tf.reduce_sum(t_scores_y)
    return t_cost

def get_indices(x,d=0):
    """
    Retorna os array de indices inteiros de [1..n] onde n é a dimensão 'd' de x
    se d não está especificado, assume d=0
    """
    return tf.to_int32(tf.range(tf.shape(x)[d]))

def gather(x, rows, cols):
    """
    Retorna x[rows, cols]
    """
    t_x_p = tf.gather_nd(x, tf.stack((rows, cols),-1))
    return t_x_p

def count_different(y,x):
    """
    retorna o numero de elementos diferentes entre y e x, ponto a ponto.
    Assume que y e x possuem o mesmo shape
    """
    return tf.reduce_sum(tf.to_int32(tf.not_equal(x, y)))

def sum_interval_features(t_x, t_s, t_f):
    """
    Sums the tensor features in t_x on intervals
    specified by index matrices t_s and t_f

    Parameters:
    t_x : tensor matrix (n_tokens,n_features)
    t_s : tensor matrix of interval start (n_intervals, 1)
    t_f : tensor matrix of interval end (n_intervals, 1)

    Returns:
    tensor matrix (n_intervals, n_features) with the sum of token
     features for each interval

    """
    n_features = tf.shape(t_x)[1]
    t_pad = tf.zeros((1,n_features))
    t_tok_features_ext = tf.concat((t_pad, t_x),axis=0)

    t_tok_csum_features = tf.cumsum(t_tok_features_ext, axis=0)
    t_end_tok = tf.gather_nd(t_tok_csum_features, t_f)
    t_start_tok = tf.gather_nd(t_tok_csum_features, t_s)
    t_interval_features = t_end_tok - t_start_tok

    return t_interval_features

def add_margin(t_scores, t_y, t_margin):
    """
    remove 't_margin' de 't_scores' nos índices especificados por 't_y'
    i.e. t_scores[t_y] -= t_margin
    """
    t_margin_values = tf.ones(tf.shape(t_y)) * t_margin
    t_margin_change = tf.scatter_nd(t_y, -t_margin_values, tf.shape(t_scores))
    t_scores_w_margin = t_scores + t_margin_change
    return t_scores_w_margin

