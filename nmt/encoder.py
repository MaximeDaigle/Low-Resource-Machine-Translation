import tensorflow as tf

# https://github.com/KonevDmitry/code_embeddings/blob/master/model/model.py

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz=32):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        gru = tf.keras.layers.LSTM(self.enc_units // 2,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        self.bidi = tf.keras.layers.Bidirectional(gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        output = self.bidi(x)

        whole_sequence_output = output[0]
        final_memory_state = tf.concat([output[1], output[3]], axis=1)
        final_carry_state = tf.concat([output[2], output[4]], axis=1)

        return whole_sequence_output, final_memory_state, final_carry_state
        # return output, stateF, stateB

    def initialize_hidden_state(self):
        init_state = [tf.zeros((self.batch_sz, self.enc_units // 2)) for i in range(2)]

        return init_state