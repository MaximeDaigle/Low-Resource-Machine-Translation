from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import sentencepiece as spm
import os
from encoder import Encoder
from decoder import DecoderNetwork
from random import shuffle
import random
import argparse


class Translator:
    def __init__(self, encoder, decoder, optimizer, name=""):
        self.decoder = decoder
        self.encoder = encoder
        self.optimizer = optimizer
        self.name = name

def load_ids(sp, filename, start_tok=True, end_tok=True):
    # <unk>=0, <s>=1, </s>=2
    with open(filename, 'r') as f_in:
        ids = list()
        for line in f_in:
            line = sp.encode_as_ids(line)
            if start_tok: line.insert(0, 1)
            if end_tok: line.append(2)
            ids.append(line)
        return ids

def max_len(tensor):
    return max(len(t) for t in tensor)


""" Training functions """

def loss_function(y_pred, y):
    # shape of y [batch_size, ty]
    # shape of y_pred [batch_size, Ty, output_vocab_size]
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    mask = tf.logical_not(tf.math.equal(y, 0))  # output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss


def train_step(input_batch, output_batch, encoder_initial_cell_state, translator, backtranslation=False):
    # initialize loss = 0
    loss = 0
    with tf.GradientTape() as tape:
        a, a_tx, c_tx = translator.encoder(input_batch, encoder_initial_cell_state)

        # [last step activations,last memory_state] of encoder passed as input to decoder Network

        # Prepare correct Decoder input & output sequence data
        decoder_input = output_batch[:, :-1]  # ignore <end>
        # compare logits with timestepped +1 version of decoder_input
        decoder_output = output_batch[:, 1:]  # ignore <start>

        # Decoder Embeddings
        decoder_emb_inp = translator.decoder.decoder_embedding(decoder_input)

        # Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
        translator.decoder.attention_mechanism.setup_memory(a)
        decoder_initial_state = translator.decoder.build_decoder_initial_state(BATCH_SIZE,
                                                                           encoder_state=[a_tx, c_tx],
                                                                           Dtype=tf.float32)
        # If backtranslation,
        # BasicDecoderOutput
        if translator.name == "en-fr":
            if backtranslation:
                outputs, _, _ = translator.decoder.decoder(decoder_emb_inp, initial_state=decoder_initial_state,
                                               sequence_length=BATCH_SIZE * [Ty_b - 1])
            else:
                outputs, _, _ = translator.decoder.decoder(decoder_emb_inp, initial_state=decoder_initial_state,
                                                           sequence_length=BATCH_SIZE * [Ty - 1])
        else:
            if backtranslation:
                outputs, _, _ = translator.decoder.decoder(decoder_emb_inp, initial_state=decoder_initial_state,
                                                      sequence_length=BATCH_SIZE * [Tx_b - 1])
            else:
                outputs, _, _ = translator.decoder.decoder(decoder_emb_inp, initial_state=decoder_initial_state,
                                                           sequence_length=BATCH_SIZE * [Tx - 1])

        logits = outputs.rnn_output
        # Calculate loss
        loss = loss_function(logits, decoder_output)

    # Returns the list of all layer variables / weights.
    variables = translator.encoder.trainable_variables + translator.decoder.trainable_variables
    # differentiate loss wrt variables
    gradients = tape.gradient(loss, variables)

    # grads_and_vars – List of(gradient, variable) pairs.
    grads_and_vars = zip(gradients, variables)
    translator.optimizer.apply_gradients(grads_and_vars)
    return loss


# RNN LSTM hidden and memory state initializer
def initialize_initial_state():
    # [num_of layers, batch, hidden]
    return [tf.zeros((BATCH_SIZE, rnn_units)), tf.zeros((BATCH_SIZE, rnn_units))]


""" Evaluation functions """
def validate(src_trg, folder, en_testset, fr_testset):
    if src_trg == "en_fr":
        testset = en_testset
        translator = en_fr
        encode_sp = sp_en
        decode_sp = sp_fr
    else:
        testset = fr_testset
        translator = fr_en
        encode_sp = sp_fr
        decode_sp = sp_en
    filename = f"/test_{src_trg}_pred_{iter+1}.txt"
    filename = folder + filename
    test_data_en = load_ids(encode_sp, testset)
    predictions = predict(test_data_en, translator)

    # prediction based on our sentence earlier
    with open(filename, 'w') as f_out:
        for i in range(len(predictions)):
            line = predictions[i, :].tolist()
            seq = list(itertools.takewhile(lambda index: index != 2, line))
            f_out.writelines([decode_sp.decode_ids(seq), '\n'])

def predict(input_sequences, translator):
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                    maxlen=Tx, padding='post')
    inp = tf.convert_to_tensor(input_sequences)

    inference_batch_size = input_sequences.shape[0]
    encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),
                                  tf.zeros((inference_batch_size, rnn_units))]

    a, a_tx, c_tx = translator.encoder(inp, encoder_initial_cell_state)

    start_tokens = tf.fill([inference_batch_size], 1)

    end_token = 2

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    decoder_input = tf.expand_dims([1] * inference_batch_size, 1)
    decoder_emb_inp = translator.decoder.decoder_embedding(decoder_input)

    decoder_instance = tfa.seq2seq.BasicDecoder(cell=translator.decoder.rnn_cell, sampler=greedy_sampler,
                                                output_layer=translator.decoder.dense_layer)
    translator.decoder.attention_mechanism.setup_memory(a)
    decoder_initial_state = translator.decoder.build_decoder_initial_state(inference_batch_size,
                                                                       encoder_state=[a_tx, c_tx],
                                                                       Dtype=tf.float32)

    # Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
    # One heuristic is to decode up to two times the source sentence lengths.
    maximum_iterations = tf.round(tf.reduce_max(Tx) * 2)

    # initialize inference decoder
    decoder_embedding_matrix = translator.decoder.decoder_embedding.variables[0]
    (first_finished, first_inputs, first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                                                                              start_tokens=start_tokens,
                                                                              end_token=end_token,
                                                                              initial_state=decoder_initial_state)
    inputs = first_inputs
    state = first_state
    predictions = np.empty((inference_batch_size, 0), dtype=np.int32)
    for j in range(maximum_iterations):
        outputs, next_state, next_inputs, finished = decoder_instance.step(j, inputs, state)
        inputs = next_inputs
        state = next_state
        outputs = np.expand_dims(outputs.sample_id, axis=-1)
        predictions = np.append(predictions, outputs, axis=-1)
    return predictions


if __name__ == "__main__":

    # Arguments passed to training script
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=1,
                        help="random seed for training")
    parser.add_argument("-n", "--num_epochs", type=int, default=50,
                        help="Number of epochs we want the model to train")
    parser.add_argument("--aligned_en", type=str, default="sub_train.lang1",
                        help="English corpus of the parallel")
    parser.add_argument("--aligned_fr", type=str, default="sub_train.lang2",
                        help="French corpus of the parallel")
    parser.add_argument("--unaligned_en", type=str, default="unaligned.en",
                        help="English monolingual corpus")
    parser.add_argument("--unaligned_fr", type=str, default="unaligned.fr",
                        help="French monolingual corpus")
    parser.add_argument("--en_bpe_model", type=str, default="en_bpe.model",
                        help="English byte-pair encoding model")
    parser.add_argument("--fr_bpe_model", type=str, default="fr_bpe.model",
                        help="French byte-pair encoding model")
    parser.add_argument("--en_testset", type=str, default="sub_test.lang1",
                        help="English test set")
    parser.add_argument("--fr_testset", type=str, default="sub_test.lang2",
                        help="French test set")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Interval between logging the training progress")
    parser.add_argument("--validation_interval", type=int, default=500,
                        help="Interval between validation")
    parser.add_argument("--result_folder", type=str, default="sentencepiece_nmt_bi_lstm",
                        help="Folder where to save checkpoints and predictions")

    args = parser.parse_args()

    epochs = args.num_epochs
    log_interval = args.log_interval
    validation_interval = args.validation_interval
    aligned_en = args.aligned_en
    aligned_fr = args.aligned_fr
    unaligned_en = args.unaligned_en
    unaligned_fr = args.unaligned_fr
    en_bpe_model = args.en_bpe_model
    fr_bpe_model = args.fr_bpe_model
    random.seed(args.seed)
    folder = args.result_folder
    en_testset = args.en_testset
    fr_testset = args.fr_testset

    """ Dataset preparation """
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(en_bpe_model)

    sp_fr = spm.SentencePieceProcessor()
    sp_fr.load(fr_bpe_model)

    # Aligned data
    data_en = load_ids(sp_en, aligned_en)
    data_fr = load_ids(sp_fr, aligned_fr)

    # Unaligned data
    unaligned_data_en = load_ids(sp_en, unaligned_en)
    shuffle(unaligned_data_en)

    unaligned_data_fr = load_ids(sp_fr, unaligned_fr)
    shuffle(unaligned_data_fr)

    # Padding
    data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding='post')
    data_fr = tf.keras.preprocessing.sequence.pad_sequences(data_fr, padding='post')
    unaligned_data_en = tf.keras.preprocessing.sequence.pad_sequences(unaligned_data_en, padding='post')
    unaligned_data_fr = tf.keras.preprocessing.sequence.pad_sequences(unaligned_data_fr, padding='post')
    unaligned_data_en = tf.convert_to_tensor(unaligned_data_en)
    unaligned_data_fr = tf.convert_to_tensor(unaligned_data_fr)

    Tx = max_len(data_en)
    Ty = max_len(data_fr)

    Tx_b = max_len(unaligned_data_en)
    Ty_b = max_len(unaligned_data_fr)

    input_vocab_size = sp_en.get_piece_size()
    output_vocab_size = sp_fr.get_piece_size()

    """ Model Parameters"""

    BATCH_SIZE = args.batch_size
    BUFFER_SIZE = len(data_en)
    steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
    embedding_dims = 256
    rnn_units = 1024
    dense_units = 1024
    Dtype = tf.float32  # used to initialize DecoderCell Zero state

    dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr)).shuffle(len(data_en)).batch(BATCH_SIZE,
                                                                                                drop_remainder=True)

    """ Creating NMT Model """
    encoderNetwork_en_fr = Encoder(input_vocab_size, embedding_dims, rnn_units, BATCH_SIZE)
    decoderNetwork_en_fr = DecoderNetwork(output_vocab_size, embedding_dims, rnn_units, dense_units, BATCH_SIZE, Tx)
    optimizer_en_fr = tf.keras.optimizers.Adam()
    en_fr = Translator(encoderNetwork_en_fr, decoderNetwork_en_fr, optimizer_en_fr, "en-fr")

    encoderNetwork_fr_en = Encoder(input_vocab_size, embedding_dims, rnn_units, BATCH_SIZE)
    decoderNetwork_fr_en = DecoderNetwork(output_vocab_size, embedding_dims, rnn_units, dense_units, BATCH_SIZE, Ty)
    optimizer_fr_en = tf.keras.optimizers.Adam()
    fr_en = Translator(encoderNetwork_fr_en, decoderNetwork_fr_en, optimizer_fr_en, "fr-en")

    """ Load checkpoint """
    # https://github.com/dhirensk/ai/blob/master/practice/english_to_french_seq2seq_tf_2_0_withattention.py

    checkpointdir = os.path.join('.', folder)
    chkpoint_prefix = os.path.join(checkpointdir, "chkpoint")
    if not os.path.exists(checkpointdir):
        os.mkdir(checkpointdir)

    checkpoint = tf.train.Checkpoint(optimizer=en_fr.optimizer, encoderNetwork=en_fr.encoder,
                                     decoderNetwork=en_fr.decoder)

    try:
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
        print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpointdir)))
    except:
        print("No checkpoint found at {}".format(checkpointdir))

    """ Train steps """
    iter = 0

    for e in range(1, epochs + 1):

        # encoder_initial_cell_state = initialize_initial_state()
        encoder_initial_cell_state = en_fr.encoder.initialize_hidden_state()
        total_loss = 0.0

        for (batch, (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
            # en-fr
            batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state, en_fr)
            total_loss += batch_loss
            if (iter + 1) % log_interval == 0:
                print("en-fr loss: {} epoch {} batch {} ".format(batch_loss.numpy(), e, batch + 1))

            # fr-en
            batch_loss = train_step(output_batch, input_batch, encoder_initial_cell_state, fr_en)
            if (iter + 1) % log_interval == 0:
                print("fr-en loss: {} epoch {} batch {} ".format(batch_loss.numpy(), e, batch + 1))

            # backtranslation en->fr->en
            i = (iter * BATCH_SIZE % len(unaligned_data_en))
            j = i + BATCH_SIZE
            if j > len(unaligned_data_en): #skip batch of different batch_size
                iter += 1
                continue
            backtranslation_input = tf.convert_to_tensor(
                predict(unaligned_data_en[i:j], en_fr))
            batch_loss = train_step(backtranslation_input, unaligned_data_en[i:j],
                                    encoder_initial_cell_state, fr_en, True)
            if (iter + 1) % log_interval == 0:
                print("en->fr->en loss: {} epoch {} batch {} ".format(batch_loss.numpy(),
                                                                      (iter // len(unaligned_data_en)) + 1, iter + 1))
            # backtranslation fr->en-fr
            backtranslation_input = tf.convert_to_tensor(
                predict(unaligned_data_fr[i:j], fr_en))
            batch_loss = train_step(backtranslation_input, unaligned_data_fr[i:j],
                                    encoder_initial_cell_state, en_fr, True)
            if (iter + 1) % log_interval == 0:
                print("fr->en->fr loss: {} epoch {} batch {} ".format(batch_loss.numpy(),
                                                                      (iter // len(unaligned_data_en)) + 1, iter + 1))

            if (iter + 1) % validation_interval == 0:
                validate("en_fr", folder, en_testset, fr_testset)
                validate("fr_en", folder, en_testset, fr_testset)
                checkpoint.save(file_prefix=chkpoint_prefix)

            iter += 1

    """ Final Translation """
    checkpoint.save(file_prefix=chkpoint_prefix)

    test_data_en = load_ids(sp_en, en_testset)
    predictions = predict(test_data_en, en_fr)

    # prediction based on our sentence earlier
    with open(folder + f"/test_en_fr_pred_final.txt", 'w') as f_out:
        for i in range(len(predictions)):
            line = predictions[i, :].tolist()
            seq = list(itertools.takewhile(lambda index: index != 2, line))
            f_out.writelines([sp_fr.decode_ids(seq), '\n'])