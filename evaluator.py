import argparse
import subprocess
import tempfile
import sentencepiece as spm
import tensorflow as tf
import os
import itertools

from nmt.nmt_seq2seq import predict, load_ids, Encoder, DecoderNetwork, max_len



def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).

    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.

    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.

    Returns: None

    """

    BATCH_SIZE = 128
    embedding_dims = 256
    rnn_units = 1024
    dense_units=1024

    # Load SentencePieceProcessors
    sp_en = spm.SentencePieceProcessor()
    sp_en.load('../model/en_bpe2.model')

    sp_fr = spm.SentencePieceProcessor()
    sp_fr.load('../model/fr_bpe2.model')

    input_vocab_size = sp_en.get_piece_size()
    output_vocab_size = sp_fr.get_piece_size()

   
    # Read input file
    tokens = load_ids(sp_en, input_file_path)
    #padded_data = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding='post')

    max_length = max_len(tokens)

    # Init model
    encoderNetwork = Encoder(input_vocab_size, embedding_dims, rnn_units, BATCH_SIZE)
    decoderNetwork = DecoderNetwork(output_vocab_size, embedding_dims, rnn_units, max_length, dense_units=dense_units, batch_size=BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()

    # Load checkpoint
    checkpointdir = "../model/sentencepiece_nmt_biLSTM_back"
    if not os.path.exists(checkpointdir):
        os.mkdir(checkpointdir)

    checkpoint = tf.train.Checkpoint(optimizer = optimizer, encoderNetwork = encoderNetwork, 
                                    decoderNetwork = decoderNetwork)

    try:
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
        print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpointdir)))
    except:
        print("No checkpoint found at {}".format(checkpointdir))

    # Start evluation
    # Start evluation
    predictions = []
    batches = [tokens[i:i+BATCH_SIZE] for i in range(0,len(tokens),BATCH_SIZE)]
    for i in batches:
        pred = predict(i, encoderNetwork, decoderNetwork, max_length, rnn_units)
        for p in pred:
            seq = list(itertools.takewhile( lambda index: index !=2, p.tolist()))
            predictions.append(sp_fr.decode_ids(seq))
            #print("Model output:", predictions[-1])

    # write answer
    with open(pred_file_path, 'w', encoding="utf-8") as anwsers:
        for pred in predictions:
            anwsers.write(pred + "\n")
    
    ##### MODIFY ABOVE #####


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.

    Returns: None

    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = out.stdout.split('\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path', help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path', help='path to input file', required=True)
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    parser.add_argument('--do-not-run-model',
                        help='will use --input-file-path as predictions, instead of running the '
                             'model on it',
                        action='store_true')

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path, args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path, args.print_all_scores)


if __name__ == '__main__':
    main()
