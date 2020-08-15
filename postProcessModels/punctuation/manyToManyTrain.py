import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential
from generateTargets import load_dataset
from sklearn.model_selection import train_test_split
import time, datetime
import pickle
import argparse


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, puctuation_encoding_size, enc_units, batch_sz=32):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bidi = tf.keras.layers.Bidirectional(gru)
        
        # one_hot vector size for punctuations after word
        mlp = tf.keras.layers.Dense(puctuation_encoding_size, activation='softmax') 
        
        # output at each timestep (direct many to many model)
        self.time_mlp = TimeDistributed(mlp)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, stateF, stateB = self.bidi(x, initial_state = hidden)
        output = self.time_mlp(output)
        return output, [stateF, stateB]

    def initialize_hidden_state(self):
        init_state = [tf.zeros((self.batch_sz, self.enc_units)) for i in range(2)]
        return init_state
        #return tf.zeros((self.batch_sz, self.enc_units))
        

#def build_model(vocab_size, embedding_dim, puctuation_encoding_size, enc_units, batch_sz=32):
#
#    model = Sequential()
#    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
#    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.enc_units,
#                                            return_sequences=True,
#                                            recurrent_initializer='glorot_uniform'))
#    model.add(TimeDistributed(Dense(puctuation_encoding_size, activation='softmax')))
#    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#    return model



class Trainner():
    
    def __init__(self, batch_size, vocab_inp_size, vocab_target_size):
        self.batch_size = batch_size
        self.embedding_dim = 256
        self.units = 1024

        # Logging
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        train_log_dir = os.path.join('logs','puct_predict',current_time,'train')
        valid_log_dir = os.path.join('logs','puct_predict',current_time,'valid') 
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        # Optimiser and loss
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(reduction='none') # one hot encoding of class

        self.encoder = Encoder(vocab_inp_size, self.embedding_dim, vocab_target_size, self.units, self.batch_size)
    
    def checkpoint(self, checkpoint_dir='./postProcessModels/checkpoints_train'):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.encoder)
        return checkpoint

    def loss_function(self, real, prediction):
        mask = tf.math.reduce_sum(real, axis=2) # return 0 if onehot is all zeros, remove padding in loss
        loss_ = self.loss_object(real, prediction)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
    
    def train_step(self, x, y, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            predictions, enc_hidden = self.encoder(x, enc_hidden)
            loss += self.loss_function(y, predictions)

        batch_loss = (loss / int(y.shape[1]))
        
        variables = self.encoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return batch_loss

    def train(self, train_dataset, steps_per_epoch, nb_epoch, checkpoint=None, valid_dataset=None):
        best_valid_score = float("inf")
        for epoch in range(nb_epoch):
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
                start = time.time()

                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if (batch) % 5000 == 0 and checkpoint is not None:
                    # Log
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('Loss', batch_loss.numpy(), step=self.optimizer.iterations)
                    print("Itr {} Loss {:.6f}".format(self.optimizer.iterations.numpy(), batch_loss))
                    print("Time taken for 1 epoch {} sec".format(time.time() - start))

                    # Validation
                    valid = ""
                    if valid_dataset is not None:
                        valid = self.valid(valid_dataset, steps_per_epoch)

                    # Save
                    checkpoint.save(file_prefix = self.checkpoint_prefix + "_valid_" + str(valid.numpy()) + "_")

    def valid(self, valid_dataset, steps_per_epoch):
        loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()

            predictions, enc_hidden = self.encoder(inp, enc_hidden)
            loss += self.loss_function(targ, predictions)

        # Log
        loss = loss / steps_per_epoch
        with self.valid_summary_writer.as_default():
            tf.summary.scalar('Loss', loss.numpy(), step=self.optimizer.iterations)
        print("Validation Loss {:.6f}\n".format(loss))

        return loss



if __name__ == "__main__":
    BATCH_SIZE = 32
    BUFFER_SIZE = 40000

    parser = argparse.ArgumentParser(description='Train a neural machine translation model')

    parser.add_argument('--src', default="./data/unaligned_nopunctuation/unaligned_en.tok", 
                        help='The source language monolingual corpus without punctuation')
    parser.add_argument('--trg', default="./data/unaligned_nopunctuation/unaligned_en_target.tok", 
                        help='The target language monolingual corpus')

    args = parser.parse_args()

    print("Generating aligned documents...")
    source_document_pth = args.src
    target_document_pth = args.trg
    input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset(source_document_pth, target_document_pth)

    checkpoint_dir = os.getcwd() + "/postProcessModels/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    with open(checkpoint_dir + "/tokenizer.pickle", 'wb') as handle:
        pickle.dump(inp_lang_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #with open(checkpoint_dir + '/tokenizer.pickle', 'rb') as handle:
    #   tokenizer = pickle.load(handle)

    print("Done generating.")

    # Split and generate dataset
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    valid_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
    valid_dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    vocab_inp_size = len(inp_lang_tokenizer.word_index)+1
    vocab_tar_size = len(targ_lang_tokenizer)

    # Start training
    trainner = Trainner(BATCH_SIZE, vocab_inp_size, vocab_tar_size)
    ckp = trainner.checkpoint(checkpoint_dir)
    trainner.train(dataset, steps_per_epoch, nb_epoch = 100000, checkpoint=ckp, valid_dataset=valid_dataset)

    

    
