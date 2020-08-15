# From a tokenized dataset. One with punctuations and the other witout.
import numpy as np
import tensorflow as tf
import os

KEEP_WORD = 0
PUNCTUATION_IDX = {"":0, ",": 1, ";": 2, ":": 3, "!":4, "?":5, ".":6, "'":7, '"':8, "(":9, ")":10, "...":11, "[":12, "]":13, "{":14, "}":15}
PUNCTUATION = ["", ",", ";", ":", "!", "?", ".", "'", '"', "(", ")", "...", "[", "]", "{", "}"]
EMB_SIZE = 64

def get_onehot_punctuation(batch_target, emb_size=EMB_SIZE):

    onehot_array = np.zeros(len(PUNCTUATION), np.uint8)
    onehot_targets = np.zeros([len(batch_target), emb_size, len(PUNCTUATION)], np.uint8)

    for ph_idx, ph in enumerate(batch_target):
        ph_array = ph.split()
        onehot_ph = []
        word_idx = 0
        onehot_word_idx = 0 # we skip over the puctuation words, this is why we have 2 indexes
        while word_idx < len(ph_array) and onehot_word_idx < emb_size:
            if (word_idx + 1 == len(ph_array)):
                onehot_targets[ph_idx, onehot_word_idx, 0] = 1
                word_idx += 1
            else:
                onehot_idx = PUNCTUATION_IDX.get(ph_array[word_idx + 1])
                if onehot_idx == None:
                    onehot_targets[ph_idx, onehot_word_idx, 0] = 1
                    word_idx += 1
                else:
                    onehot_targets[ph_idx, onehot_word_idx, onehot_idx] = 1
                    word_idx += 2
            onehot_word_idx += 1

    return onehot_targets

def tokenize(lang, maxlen):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=maxlen)

    return tensor, lang_tokenizer

def load_dataset(source_path, target_path, start=0, finish=None, maxlen = 64):
    # creating cleaned input, output pairs
    inp_lang, targ_lang = create_dataset(source_path, target_path, start, finish)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang, maxlen)
    target_tensor = arg = get_onehot_punctuation(targ_lang, maxlen)

    return input_tensor, target_tensor, inp_lang_tokenizer, PUNCTUATION

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(source_path, target_path, start=0, finish=None):
    source_lines = open(source_path, encoding='UTF-8').read().strip().split('\n')
    target_lines = open(target_path, encoding='UTF-8').read().strip().split('\n')

    finish = len(source_lines) if finish == None else finish
    assert len(source_lines) == len(target_lines), "Not the same number of lines in target and input files."

    source = source_lines[start:finish]
    target = target_lines[start:finish]

    return source, target

def add_punctuation(input_text, model_output):
    """
    Used to generate text from one-hot encoded model output. 
    We need the original text without punctuation to add it.
    """
    output = []
    for ph_idx, ph in enumerate(input_text):
        output_ph = []
        for word_idx, word in enumerate(ph): # ph.split()
            output_ph.append(word)
            arg_max = tf.math.argmax(model_output[ph_idx, word_idx]).numpy()
            if arg_max != 0:
                output_ph.append(PUNCTUATION[arg_max])
        output.append(" ".join(output_ph))
    
    return output

if __name__ == "__main__":
    
    punc = ["Hello there , world !", "", "HI"]
    no_punc = ["Hello there world", "", "HI"]
    targets = get_onehot_punctuation(punc, 10)

    # test that the onehot encoding and punctuation adding works
    assert punc == add_punctuation(no_punc, targets), "Adding puctuation does not work!"
    print(targets, targets.shape)

    source_document_pth = os.getcwd() + "/data/unaligned_nopunctuation/unaligned_en.tok"
    target_document_pth = os.getcwd() + "/data/unaligned_nopunctuation/unaligned_en_target.tok"
    input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset(source_document_pth, target_document_pth, 0, 1000)
    pass


