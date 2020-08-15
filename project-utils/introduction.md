# Low-Resource Machine Translation Introduction

Note: this is a recap of some of the presentation slides available [here](https://github.com/mila-iqia/ift6759/blob/master/projects/project2/low_resource_machine_translation_project.pdf).

## Goal

Translate non-formatted English (i.e., English that is all lower case and where most of the punctuation
has been removed) into French. This happens in a low-resource scenario where you have access to only a
limited amount of parallel examples. Plus, you will have access to unaligned monolingual data.

## Provided Data

We will provide all teams with the following:
* 11k parallel examples (non-formatted English => French). This data has been already tokenized.
* 474k monolingual examples in English. This data is **not** tokenized.
* 474k monolingual examples in French. This data is **not** tokenized.

The data can be found on Helios: `/project/cq-training-1/project2/data`.

**IMPORTANT NOTE: You are NOT allowed to use any additional data.** Due to this imposed limitation,
you are also **not** allowed to use models (or embeddings) that are pre-trained on 3rd-party data.
Teams ignoring this directive will get **zero** in the evaluation.

## Provided Scripts

We also provide to you the scripts we used to tokenize the data (`tokenizer.py`) and strip the
punctuation from the data (`punctuation_remover.py`). In order to use them, you need to install the
related dependencies first:

    pip install -r requirements.txt

### Tokenizer

The tokenizer script is based on `spacy`. It can be used with various options (see `--help`).
For example, to perform tokenization on French data, keeping the case:

    python tokenizer.py --input train.txt --output output_folder --lang fr --keep-case --keep-empty-lines

Note the option `--keep-empty-lines`. You may want to use this if you want to maintain the alignment
between the input files and the output files (in case empty lines are present).

### Punctuation Remover

This script can be used on already tokenized data (using the previous script) to remove the
punctuation the same way as we did for the English language in the parallel data.

    python punctuation_remover.py --input train.tok --output output_folder

## Evaluation

Refer to the [evaluation page](evaluation.md).