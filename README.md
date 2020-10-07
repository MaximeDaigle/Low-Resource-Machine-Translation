# Low Resource Machine Translation

### [[Paper]](https://github.com/MaximeDaigle/Low-Resource-Machine-Translation/blob/master/Low%20Resource%20Machine%20Translation.pdf)

![embedding](https://github.com/MaximeDaigle/Low-Resource-Machine-Translation/blob/master/nmt_embedding.png)

# To evaluate
Please modify evaluate_on_slurm.sh so that the evaluator.py arguments point to the proper source and target files:
```
./evaluator.py --input-file-path "/path/to/english.txt" --target-file-path "/path/to/French.txt"
```

Then, to evaluate simply run:
```
sbatch evaluate_on_slurm.sh
```

We are using a specific python environment. Make sure to have access to /project/cq-training-1/project2/submissions/team07/low_env
and /project/cq-training-1/project2/submissions/team07/model since these paths are hard codded.

# To train

Most of the preprocessing was done in the notebooks folder.

Our evaluation model is part of the NMT_networks_seq2seq_nmt.ipynb notebook. To see how to train refer to notebook.

### Training undreamt-tf
When training the semi-supervised model undreamt:
```
python ./undreamt-tf/train.py \
    --src /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/en_corpus.txt.atok \
    --trg /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/fr_corpus.txt.atok \
    --src_embeddings /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/SRC_MAPPED.EMB \
    --trg_embeddings /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/TRG_MAPPED.EMB \
    --src2trg /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/sub_train.lang1.atok \
            /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/sub_train.lang2.atok \
    --save semi-supervised --iterations 30000 --cache_parallel 1000000 \
    --validation /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/sub_test.lang1 \
                /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/sub_test.lang2 \
    --validation_directions src2trg --log_interval 1000
```

## Training Bi-LSTM with Backtranslation
We trained the model with all the default parameters:
```
python ./nmt/train.py
```

However, here is an example where the model is trained without default parameters:
```
python ./nmt/train.py \
  --seed 213 \
  --num_epochs 100 \
  --aligned_en \path\parallel_en.txt \
  --aligned_fr \path\parallel_fr.txt \
  --unaligned_en \path\monolingual_en.txt \
  --unaligned_fr \path\monolingual_fr.txt \
  --en_bpe_model \path\model_en \
  --fr_bpe_model \path\model_fr \
  --en_testset \path\testset_english.txt \
  --fr_testset \path\testset_french.txt \
  --batch_size 64 \
  --log_interval 100 \
  --validation_interval 1000 \
  --result_folder result
```


### Trainning post-processing many to many punctuation
Simply run the python file as such:
```
python ./postProcessModels/punctuation/manyToManyTrain.py --src "tokenized/without/punctuation.tk" --trg "tokenized/with/punctuation.tk"
```
Note: Make sure to use the tokenizer and punctuation remover from the project-utils folder before passing these target and source files.
