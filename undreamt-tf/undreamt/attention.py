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


class GlobalAttention(tf.keras.Model):
    def __init__(self, dim, alignment_function='general'):
        super(GlobalAttention, self).__init__()
        self.alignment_function = alignment_function
        if self.alignment_function == 'general':
            # self.linear_align = nn.Linear(dim, dim, bias=False)
            self.linear_align = tf.keras.layers.Dense(dim, input_shape=(dim,),use_bias=False)
        elif self.alignment_function != 'dot':
            raise ValueError('Invalid alignment function: {0}'.format(alignment_function))
        # #self.softmax = nn.Softmax(dim=1)
        # self.softmax = tf.nn.softmax(axis=1)
        #self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_context = tf.keras.layers.Dense(dim, input_shape=(dim,),use_bias=False)
        #self.linear_query = nn.Linear(dim, dim, bias=False)
        self.linear_query = tf.keras.layers.Dense(dim, input_shape=(dim,),use_bias=False)
        # #self.tanh = nn.Tanh()
        # self.tanh = tf.keras.activations.tanh()

    def call(self, query, context, mask):
        # query: batch*dim
        # context: length*batch*dim
        # ans: batch*dim

        #context_t = context.transpose(0, 1)  # batch*length*dim
        context_t = tf.transpose(context, perm=[1, 0, 2]) # batch*length*dim

        # Compute alignment scores
        q = query if self.alignment_function == 'dot' else self.linear_align(query)
        #align = context_t.bmm(q.unsqueeze(2)).squeeze(2)  # batch*length
        align = tf.squeeze(tf.matmul(context_t, tf.expand_dims(q, 2)), [2])  # batch*length
        # Mask alignment scores
        if mask is not None:
            #align.data.masked_fill_(mask, -float('inf'))
            align = tf.where(mask, -float('inf'), align)

        # Compute attention from alignment scores
        attention = tf.nn.softmax(align, axis=1)  # batch*length

        # Computed weighted context
        #weighted_context = attention.unsqueeze(1).bmm(context_t).squeeze(1)  # batch*dim
        weighted_context = tf.squeeze(tf.matmul(tf.expand_dims(attention, 1), context_t), [1])  # batch*dim

        # Combine context and query
        return tf.keras.activations.tanh(self.linear_context(weighted_context) + self.linear_query(query))