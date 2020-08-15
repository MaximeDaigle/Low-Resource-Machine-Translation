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

import tensorflow as tf
import tensorflow.keras.layers as nn
import sys
from undreamt import data
from undreamt.stacked_gru import GRU



class RNNEncoder(tf.keras.Model):
    def __init__(self, embedding_size, hidden_size, bidirectional=False, nb_layers=1, dropout=0):
        super(RNNEncoder, self).__init__()
        if bidirectional and hidden_size % 2 != 0:
            raise ValueError('The hidden dimension must be even for bidirectional encoders')
        self.directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.nb_layers = nb_layers
        self.hidden_size = hidden_size // self.directions
        self.special_embeddings = nn.Embedding(data.SPECIAL_SYMBOLS+1, embedding_size)
        self.rnn = GRU(num_units=self.hidden_size, bidirectional=bidirectional, num_layers=nb_layers, dropout=dropout)
        #self.rnn = tf.keras.layers.Bidirectional(
        #    tf.keras.layers.GRU(self.hidden_size, num_layers=nb_layers, dropout=dropout))

    def call(self, ids, lengths, word_embeddings, hidden):
        sorted_lengths = sorted(lengths, reverse=True)
        is_sorted = sorted_lengths == lengths
        is_varlen = sorted_lengths[0] != sorted_lengths[-1]
        if tf.reduce_sum(hidden) != 0:
            print('****need to pass hidden as initial_state in GRU****')
            sys.exit(-1)
        if not is_sorted:
            true2sorted = sorted(range(len(lengths)), key=lambda x: -lengths[x])
            sorted2true = sorted(range(len(lengths)), key=lambda x: true2sorted[x])
            ids = tf.stack([ids[:, i] for i in true2sorted], axis=1)
            lengths = [lengths[i] for i in true2sorted]
        embeddings = word_embeddings(data.word_ids(ids)) + self.special_embeddings(data.special_ids(ids))
        # if is_varlen:
        #     embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths)
        embeddings = tf.transpose(embeddings, perm=[1,0,2])
        embeddings_mask = tf.transpose(word_embeddings.compute_mask(data.word_ids(ids)), perm=[1,0])
        # if (embeddings_mask.numpy().sum(axis=1)==0).any():
        #     print(embeddings_mask.numpy().sum(axis=1))
        output, hidden = self.rnn(embeddings, mask=embeddings_mask)
        hidden = tf.convert_to_tensor(hidden)
        output = tf.transpose(output, perm=[1, 0, 2])
        if self.bidirectional:
            hidden= tf.squeeze(hidden, [1])
        if not self.bidirectional:
            print('****Encoder not bidirectional was not Tested****')
        # if is_varlen: # TODO didn't touch that possibility
        #     output = nn.utils.rnn.pad_packed_sequence(output)[0]
        if not is_sorted: # TODO didn't touch that possibility
            hidden = tf.stack([hidden[:, i, :] for i in sorted2true], axis=1)
            output = tf.stack([output[:, i, :] for i in sorted2true], axis=1)
        # print(output.shape, hidden.shape)  # at the end, want torch.Size([3, 5, 600]), torch.Size([2, 5, 600])
        return hidden, output

    def initial_hidden(self, batch_size):
        return tf.Variable(tf.zeros([self.nb_layers*self.directions, batch_size, self.hidden_size]), trainable=False)