import numpy as np
import tensorflow as tf
from manyToManyTrain import Trainner
from generateTargets import PUNCTUATION, add_punctuation
import os
import pickle

if __name__ == "__main__":
    BATCH_SIZE=1
    checkpoint_dir = os.getcwd() + "/postProcessModels/best-valid"
    with open(checkpoint_dir + '/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    vocab_inp_size = len(tokenizer.word_index)+1
    vocab_tar_size = len(PUNCTUATION)
    trainner = Trainner(BATCH_SIZE, vocab_inp_size, vocab_tar_size)

    checkpoint = tf.train.Checkpoint(model=trainner.encoder, optimizer=trainner.optimizer)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    lang = [#['This', 'is', 'just', 'an', 'example'],
            'Therefore my feeling here is that we need to carry out a genuine technical and financial review of the system and take to task those who have been managing this project'.split()]
    token_lang = tokenizer.texts_to_sequences(lang)
    #token_lang = np.array(token_lang).

    enc_hidden = trainner.encoder.initialize_hidden_state()
    predictions, enc_hidden = trainner.encoder(tf.convert_to_tensor(token_lang), enc_hidden)

    print(add_punctuation(lang, predictions))

