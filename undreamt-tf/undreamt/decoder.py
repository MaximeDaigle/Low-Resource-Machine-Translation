# Copyright (C) 2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from undreamt import data
from undreamt.attention import GlobalAttention
from undreamt.stacked_gru import GRU

# import data
# from attention import GlobalAttention

import tensorflow as tf
import tensorflow.keras.layers as nn

class RNNAttentionDecoder(tf.keras.Model):
    def __init__(self, embedding_size, hidden_size, nb_layers=1, dropout=0, input_feeding=True):
        super(RNNAttentionDecoder, self).__init__()
        self.nb_layers = nb_layers
        self.hidden_size = hidden_size
        self.special_embeddings = nn.Embedding(data.SPECIAL_SYMBOLS+1, embedding_size)
        self.attention = GlobalAttention(hidden_size, alignment_function='general')
        self.input_feeding = input_feeding
        #self.stacked_rnn = StackedGRU(hidden_size, nb_layers=nb_layers, dropout=dropout)
        self.stacked_rnn = GRU(nb_layers, hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def call(self, ids, lengths, word_embeddings, hidden, context, context_mask, prev_output, generator):
        embeddings = word_embeddings(data.word_ids(ids)) + self.special_embeddings(data.special_ids(ids))
        output = prev_output
        scores = []
        for emb in embeddings:
            if self.input_feeding:
                input = tf.concat([emb, output], 1)
            else:
                input = emb
            # output, hidden = self.stacked_rnn(input, hidden)

            input = tf.expand_dims(input, 1)
            output, hidden = self.stacked_rnn(input, initial_state=hidden)
            hidden = tf.convert_to_tensor(hidden)
            output, hidden = tf.squeeze(output, [1]), tf.squeeze(hidden, [1])

            output = self.attention(output, context, context_mask)
            output = self.dropout(output)
            scores.append(generator(output))
        return tf.stack(scores), hidden, output

    def initial_output(self, batch_size):
        return tf.Variable(tf.zeros([batch_size, self.hidden_size]), trainable=False)


# # Based on OpenNMT-py
# class StackedGRU(tf.keras.Model):
#     def __init__(self, hidden_size, nb_layers, dropout):
#         super(StackedGRU, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.num_layers = nb_layers
#         self.stacked_layers = []
#         for i in range(nb_layers):
#             self.stacked_layers.append(nn.GRUCell(hidden_size))
#             input_size = hidden_size
#
#
#     def call(self, input, hidden):
#         h_1 = []
#         for i, layer in enumerate(self.stacked_layers):
#             h_1_i = layer(input, hidden[i])
#             input = h_1_i
#             if i + 1 != self.num_layers:
#                 input = self.dropout(input)
#             h_1 += [h_1_i]
#         h_1 = tf.stack(h_1)
#         return input, h_1

if __name__ == "__main__":
    dropout = 0.2
    input_size = 600
    hidden_size=600
    nb_layers=2
    batch_size = 5
    # m = StackedGRU(hidden_size=hidden_size, nb_layers=nb_layers, dropout=0.2)
    from stacked_gru import GRU
    m = GRU(nb_layers, hidden_size, dropout=dropout)
    input = tf.random.uniform(shape=[batch_size,900])
    hidden = tf.random.uniform(shape=[nb_layers,batch_size,hidden_size])
    input = tf.expand_dims(input,1)
    output, h_1 = m(input,initial_state=hidden)
    h_1 = tf.convert_to_tensor(h_1)
    output, h_1 =  tf.squeeze(output,[1]), tf.squeeze(h_1,[1])
    print(output.shape, h_1.shape)
    output_size =[5, 600]
    h_1_size = [2, 5, 600]