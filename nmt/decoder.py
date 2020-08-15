import tensorflow as tf
import tensorflow_addons as tfa

# https://github.com/KonevDmitry/code_embeddings/blob/master/model/model.py
# DECODER
class DecoderNetwork(tf.keras.Model):
    def __init__(self, output_vocab_size, embedding_dims, rnn_units, dense_units, BATCH_SIZE, Tx):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims)
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.dense_units = dense_units
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(dense_units, None, BATCH_SIZE * [Tx])
        self.rnn_cell = self.build_rnn_cell(BATCH_SIZE)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, units, memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory=memory,
                                          memory_sequence_length=memory_sequence_length)
        # return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell
    def build_rnn_cell(self, batch_size):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=self.dense_units)
        return rnn_cell

    def build_decoder_initial_state(self, batch_size, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size,
                                                                dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state