from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
import itertools
from pickle import load
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from pickle import load
from numpy import array
from numpy import argmax
import tensorflow as tf
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import numpy as np
import sentencepiece as spm
import os

# do_train = False
# epochs = 30
# 
# sp_en = spm.SentencePieceProcessor()
# sp_en.load('en_bpe.model')
# 
# sp_fr = spm.SentencePieceProcessor()
# sp_fr.load('fr_bpe.model')

def load_ids(sp, filename, start_tok=True, end_tok=True):
    #<unk>=0, <s>=1, </s>=2
    with open(filename, 'r', encoding="utf-8") as f_in:
        ids= list()
        for line in f_in:
            line= sp.encode_as_ids(line)
            if start_tok: line.insert(0,1)
            if end_tok: line.append(2)
            ids.append(line)
        return ids

# data_en = load_ids(sp_en, "sub_train.lang1")
# print(data_en[0])
# print(sp_en.decode_ids(data_en[0]))
# 
# data_fr = load_ids(sp_fr, "sub_train.lang2")
# print(data_fr[0])
# print(sp_fr.decode_ids(data_fr[0]))
# 
# data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')
# data_fr = tf.keras.preprocessing.sequence.pad_sequences(data_fr,padding='post')

# for i in range(5): print(data_en[:i])

def max_len(tensor):
    #print( np.argmax([len(t) for t in tensor]))
    return max( len(t) for t in tensor)

"""## Model Parameters"""

# X_train,  X_test, Y_train, Y_test = train_test_split(data_en,data_fr,test_size=0.2)
# X_train, Y_train = data_en, data_fr
# BATCH_SIZE = 128
# BUFFER_SIZE = len(X_train)
# steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
# embedding_dims = 256
# rnn_units = 1024
#dense_units = 1024
# Dtype = tf.float32   #used to initialize DecoderCell Zero state

"""## Dataset Prepration"""
"""
Tx = max_len(data_en)
Ty = max_len(data_fr)  

input_vocab_size = sp_en.get_piece_size()
output_vocab_size = sp_fr.get_piece_size()

print(input_vocab_size, output_vocab_size)
dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
example_X, example_Y = next(iter(dataset))
print(example_X.shape) 
print(example_Y.shape)
"""
"""## Defining NMT Model"""

# https://github.com/KonevDmitry/code_embeddings/blob/master/model/model.py

#ENCODER
class EncoderNetwork(tf.keras.Model):
    def __init__(self,input_vocab_size,embedding_dims, rnn_units, bidirectional=True):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, 
                                                     return_state=True )


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

        whole_sequence_output=output[0]
        final_memory_state=tf.concat([output[1], output[3]], axis=1)
        final_carry_state=tf.concat([output[2], output[4]], axis=1)

        return whole_sequence_output, final_memory_state, final_carry_state
        # return output, stateF, stateB

    def initialize_hidden_state(self):
        init_state = [tf.zeros((self.batch_sz, self.enc_units //2 )) for i in range(2)]
        
        return init_state

#DECODER
class DecoderNetwork(tf.keras.Model):
    def __init__(self,output_vocab_size, embedding_dims, rnn_units, max_len, dense_units=1024, batch_size=128, ):
        super().__init__()
        self.dense_units = dense_units
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims) 
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(dense_units,None,batch_size*[max_len])
        self.rnn_cell =  self.build_rnn_cell(batch_size)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, units,memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory = memory, 
                                          memory_sequence_length=memory_sequence_length)
        #return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell  
    def build_rnn_cell(self, batch_size ):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=self.dense_units)
        return rnn_cell
    
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state

# encoderNetwork = EncoderNetwork(input_vocab_size,embedding_dims, rnn_units)
"""
encoderNetwork = Encoder(input_vocab_size, embedding_dims, rnn_units, BATCH_SIZE)
decoderNetwork = DecoderNetwork(output_vocab_size, embedding_dims, rnn_units)
optimizer = tf.keras.optimizers.Adam()
"""

"""## Initializing Training functions"""

def loss_function(y_pred, y):
   
    #shape of y [batch_size, ty]
    #shape of y_pred [batch_size, Ty, output_vocab_size] 
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss


def train_step(input_batch, output_batch,encoder_initial_cell_state):
    #initialize loss = 0
    loss = 0
    with tf.GradientTape() as tape:
        # encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)

        # a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
        #                                                 initial_state =encoder_initial_cell_state)
        a, a_tx, c_tx = encoderNetwork(input_batch, encoder_initial_cell_state)

        #[last step activations,last memory_state] of encoder passed as input to decoder Network
        
         
        # Prepare correct Decoder input & output sequence data
        decoder_input = output_batch[:,:-1] # ignore <end>
        #compare logits with timestepped +1 version of decoder_input
        decoder_output = output_batch[:,1:] #ignore <start>


        # Decoder Embeddings
        decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

        #Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
        decoderNetwork.attention_mechanism.setup_memory(a)
        decoder_initial_state = decoderNetwork.build_decoder_initial_state(BATCH_SIZE,
                                                                           encoder_state=[a_tx, c_tx],
                                                                           Dtype=tf.float32)
        
        #BasicDecoderOutput        
        outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
                                               sequence_length=BATCH_SIZE*[Ty-1])

        logits = outputs.rnn_output
        #Calculate loss

        loss = loss_function(logits, decoder_output)

    #Returns the list of all layer variables / weights.
    variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables  
    # differentiate loss wrt variables
    gradients = tape.gradient(loss, variables)

    #grads_and_vars – List of(gradient, variable) pairs.
    grads_and_vars = zip(gradients,variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss

#RNN LSTM hidden and memory state initializer
def initialize_initial_state():
    # [num_of layers, batch, hidden]
    return [tf.zeros((BATCH_SIZE, rnn_units)), tf.zeros((BATCH_SIZE, rnn_units))]

"""## Training

Load checkpoint
"""

# https://github.com/dhirensk/ai/blob/master/practice/english_to_french_seq2seq_tf_2_0_withattention.py
"""
checkpointdir = os.path.join('.',"sentencepiece_nmt_bi_lstm")
chkpoint_prefix = os.path.join(checkpointdir, "chkpoint")
if not os.path.exists(checkpointdir):
    os.mkdir(checkpointdir)

checkpoint = tf.train.Checkpoint(optimizer = optimizer, encoderNetwork = encoderNetwork, 
                                 decoderNetwork = decoderNetwork)

try:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
    print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpointdir)))
except:
    print("No checkpoint found at {}".format(checkpointdir))
"""

"""## Evaluation"""

def predict(input_sequences, encoderNetwork, decoderNetwork, max_length, rnn_units):
    #In this section we evaluate our model on a raw_input, for this the entire sentence has to be passed
    #through the length of the model, for this we use greedsampler to run through the decoder
    #and the final embedding matrix trained on the data is used to generate embeddings
    # input_raw='▁i ▁agree ▁that ▁we ▁need ▁an ▁ambitious ▁social ▁agenda ▁which ▁will ▁include ▁combating ▁poverty ▁and ▁social ▁exclusion'
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                    maxlen=max_length, padding='post')
    inp = tf.convert_to_tensor(input_sequences)
    # print(inp.shape)
    inference_batch_size = input_sequences.shape[0]
    encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),
                                tf.zeros((inference_batch_size, rnn_units))]
    # encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
    # a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp,
    #                                                 initial_state =encoder_initial_cell_state)
    a, a_tx, c_tx = encoderNetwork(inp,encoder_initial_cell_state)
    # print('a_tx :',a_tx.shape)
    # print('c_tx :', c_tx.shape)

    start_tokens = tf.fill([inference_batch_size], 1)

    end_token = 2

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    decoder_input = tf.expand_dims([1]* inference_batch_size,1)
    decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

    decoder_instance = tfa.seq2seq.BasicDecoder(cell = decoderNetwork.rnn_cell, sampler = greedy_sampler,
                                                output_layer=decoderNetwork.dense_layer)
    decoderNetwork.attention_mechanism.setup_memory(a)
    #pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
    # print("decoder_initial_state = [a_tx, c_tx] :",np.array([a_tx, c_tx]).shape)
    decoder_initial_state = decoderNetwork.build_decoder_initial_state(inference_batch_size,
                                                                    encoder_state=[a_tx, c_tx],
                                                                    Dtype=tf.float32)
    # print("\nCompared to simple encoder-decoder without attention, the decoder_initial_state \
    #  is an AttentionWrapperState object containing s_prev tensors and context and alignment vector \n ")
    # print("decoder initial state shape :",np.array(decoder_initial_state).shape)
    # print("decoder_initial_state tensor \n", decoder_initial_state)

    # Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
    # One heuristic is to decode up to two times the source sentence lengths.
    maximum_iterations = tf.round(tf.reduce_max(max_length) * 2)

    #initialize inference decoder
    decoder_embedding_matrix = decoderNetwork.decoder_embedding.variables[0] 
    (first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                                start_tokens = start_tokens,
                                end_token=end_token,
                                initial_state = decoder_initial_state)
    #print( first_finished.shape)
    # print("\nfirst_inputs returns the same decoder_input i.e. embedding of  <start> :",first_inputs.shape)
    # print("start_index_emb_avg ", tf.reduce_sum(tf.reduce_mean(first_inputs, axis=0))) # mean along the batch

    inputs = first_inputs
    state = first_state  
    predictions = np.empty((inference_batch_size,0), dtype = np.int32)                                                                             
    for j in range(maximum_iterations):
        outputs, next_state, next_inputs, finished = decoder_instance.step(j,inputs,state)
        inputs = next_inputs
        state = next_state
        outputs = np.expand_dims(outputs.sample_id,axis = -1)
        predictions = np.append(predictions, outputs, axis = -1)
    return predictions

"""## Final Translation"""

"""Train steps"""
"""
for i in range(1, epochs+1):

    # encoder_initial_cell_state = initialize_initial_state()
    encoder_initial_cell_state = encoderNetwork.initialize_hidden_state()
    total_loss = 0.0

    for ( batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
        total_loss += batch_loss
        if (batch+1)%5 == 0:
            print("total loss: {} epoch {} batch {} ".format(batch_loss.numpy(), i, batch+1))
    checkpoint.save(file_prefix = chkpoint_prefix)


    # test_data_en=load_data("sub_test.lang1.atok", end_tok=False)
    test_data_en=load_ids(sp_en, "sub_test.lang1")
    predictions = predict(test_data_en)

    line_ = list(itertools.takewhile( lambda index: index !=2, predictions[10].tolist()))
    # print(sp_en.decode_ids(test_data_en[10]))
    # print(sp_fr.decode_ids(line_))

    #prediction based on our sentence earlier
    with open(f"sentencepiece_nmt_bi_gru/test_fr_pred_{i}.txt", 'w') as f_out:
        for i in range(len(predictions)):
            line = predictions[i,:].tolist()
            seq = list(itertools.takewhile( lambda index: index !=2, line))
            f_out.writelines([sp_fr.decode_ids(seq), '\n'])
"""

def predict_beam(input_sequences, decoderNetwork):       

    """### Inference using Beam Search with beam_width = 3"""

    beam_width = 3
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                    maxlen=Tx, padding='post')
    inp = tf.convert_to_tensor(input_sequences)
    #print(inp.shape)
    inference_batch_size = input_sequences.shape[0]
    encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),
                                tf.zeros((inference_batch_size, rnn_units))]
    a, a_tx, c_tx = encoderNetwork(inp,encoder_initial_cell_state)
    #pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
    s_prev = [a_tx, c_tx]

    start_tokens = tf.fill([inference_batch_size],1)
    end_token = 2

    decoder_input = tf.expand_dims([1]* inference_batch_size,1)
    decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)


    #From official documentation
    #NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:

    #The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
    #The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
    #The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.
    encoder_memory = tfa.seq2seq.tile_batch(a, beam_width)
    decoderNetwork.attention_mechanism.setup_memory(encoder_memory)
    print("beam_with * [batch_size, Tx, rnn_units] :  3 * [2, Tx, rnn_units]] :", encoder_memory.shape)
    #set decoder_inital_state which is an AttentionWrapperState considering beam_width
    decoder_initial_state = decoderNetwork.rnn_cell.get_initial_state(batch_size = inference_batch_size* beam_width,dtype = Dtype)
    encoder_state = tfa.seq2seq.tile_batch(s_prev, multiplier=beam_width)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 

    decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoderNetwork.rnn_cell,beam_width=beam_width,
                                                    output_layer=decoderNetwork.dense_layer)


    # Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
    # One heuristic is to decode up to two times the source sentence lengths.
    maximum_iterations = tf.round(tf.reduce_max(Tx) * 2)

    #initialize inference decoder
    decoder_embedding_matrix = decoderNetwork.decoder_embedding.variables[0] 
    (first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                                start_tokens = start_tokens,
                                end_token=end_token,
                                initial_state = decoder_initial_state)
    #print( first_finished.shape)
    print("\nfirst_inputs returns the same decoder_input i.e. embedding of  <start> :",first_inputs.shape)

    inputs = first_inputs
    state = first_state  
    predictions = np.empty((inference_batch_size, beam_width,0), dtype = np.int32)
    beam_scores =  np.empty((inference_batch_size, beam_width,0), dtype = np.float32)                                                                            
    for j in range(maximum_iterations):
        beam_search_outputs, next_state, next_inputs, finished = decoder_instance.step(j,inputs,state)
        inputs = next_inputs
        state = next_state
        outputs = np.expand_dims(beam_search_outputs.predicted_ids,axis = -1)
        scores = np.expand_dims(beam_search_outputs.scores,axis = -1)
        predictions = np.append(predictions, outputs, axis = -1)
        beam_scores = np.append(beam_scores, scores, axis = -1)
    print(predictions.shape) 
    print(beam_scores.shape)
