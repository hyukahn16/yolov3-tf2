import tensorflow as tf

CHAN_FIRST = [0, 3, 1, 2] # Used to change image TO channel first
CHAN_LAST = [0, 2, 3, 1] # Used to change image TO channel last

def channel_to_first(data):
    return tf.transpose(data, CHAN_FIRST)

def channel_to_last(data):
    return tf.transpose(data, CHAN_LAST)