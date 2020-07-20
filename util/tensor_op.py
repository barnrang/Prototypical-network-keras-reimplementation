import tensorflow as tf

def slice_tensor_and_sum(x, way=20):
    sliced = tf.split(x, num_or_size_splits=way,axis=0)
    return tf.reduce_mean(sliced, axis=1)

def reduce_tensor(x):
    return tf.reduce_mean(x, axis=1)

def reshape_query(x):
    return tf.reshape(x, [-1, tf.shape(x)[-1]])
