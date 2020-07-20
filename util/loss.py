import tensorflow as tf

def proto_dist(x):
    feature, pred = x
    pred_dist = tf.reduce_sum(pred ** 2, axis=1, keepdims=True)
    feature_dist = tf.reduce_sum(feature ** 2, axis=1, keepdims=True)
    dot = tf.matmul(pred, tf.transpose(feature))
    return tf.nn.softmax(-(tf.sqrt(pred_dist + tf.transpose(feature_dist) - 2 * dot)))

#def prior_dist(x):
#    feature, pred = x
#def prior_loss(target, pred):
#    return tf.losses.softmax_cross_entropy(target, -pred)
#
#def prior_acc(target, pred):
#

