import tensorflow as tf

class DummyModel(tf.keras.Model):

    def __init__(self, target_time_offsets):
        super(DummyModel, self).__init__()

    def call(self, x):
        return x

    def load_config(self, user_config):
        pass
