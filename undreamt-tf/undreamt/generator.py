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

import tensorflow as tf
import tensorflow.keras.layers as layers


class EmbeddingGenerator(tf.keras.Model):
    def __init__(self, hidden_size, embedding_size):
        super(EmbeddingGenerator, self).__init__()
        self.hidden2embedding = layers.dense(hidden_size, embedding_size)
        self.special_out = layers.dense(embedding_size, data.SPECIAL_SYMBOLS, bias=False)

    def call(self, hidden, embeddings):
        emb = self.hidden2embedding(hidden)
        word_scores = tf.matmul(emb, embeddings.weight[1:, :])
        #word_scores = F.linear(emb, embeddings.weight[1:, :])
        special_scores = self.special_out(emb)
        scores = tf.concat((special_scores, word_scores), axis=1)
        # return tf.nn.log_softmax(scores, axis=1)
        return scores

    def output_classes(self):
        return None


class WrappedEmbeddingGenerator(tf.keras.Model):
    def __init__(self, embedding_generator, embeddings):
        super(WrappedEmbeddingGenerator, self).__init__()
        self.embedding_generator = embedding_generator
        self.embeddings = embeddings

    def call(self, hidden):
        return self.embedding_generator(hidden, self.embeddings)

    def output_classes(self):
        return self.embeddings.weight.data.size()[0] + data.SPECIAL_SYMBOLS - 1


class LinearGenerator(tf.keras.Model):
    def __init__(self, hidden_size, vocabulary_size, bias=True):
        super(LinearGenerator, self).__init__()
        self.out = layers.Dense(data.SPECIAL_SYMBOLS + vocabulary_size, use_bias=bias)
        # self.logsoftmax = tf.nn.log_softmax(axis=1)

    def call(self, hidden):
        # return tf.nn.log_softmax(self.out(hidden), axis=1)
        return self.out(hidden)

    def output_classes(self):
        return self.out.units