# Model Evaluation Guidelines

Once again, we provide a script ([``evaluator.py``](evaluator.py)) in this folder for the evaluation
of your submitted model. This script is needed to:
1) score your model (on any data set you want) with the 'official' BLEU metric; and
2) call your model to generate predictions on our (blind) test set so that we can rank it.

## Scoring your model

You can use the script to score your model. The script will call
[sacreBLEU](https://github.com/mjpost/sacreBLEU) with the correct options.

The quickest way to score your model output is to write the predictions to a file (say
`predictions.txt`) and the targets to another file (say `targets.txt`) and run:

    python evaluator.py --input-file-path predictions.txt --target-file-path targets.txt --do-not-run-model

Note the flag `--do-not-run-model` which means that the input file will be considered as
predictions (instead of as inputs for your model).

If you want to call your models directly inside the evaluator script instead, you should not use
the flag `--do-not-run-model` and you should pass the input data (say `inputs.txt`) to the
flag `--input-file-path`. For example:

    python evaluator.py --input-file-path inputs.txt --target-file-path targets.txt

In this case, the script will call your model to generate prediction on the input file (`inputs.txt`),
it will store the predictions in a temporary file, and it will use these predictions to compute the
BLEU score. Note that for this to work, you will need to include (in the `evaluator.py` script) the
code to run your model. See next sections for more details.

## Modifying the script

In order to allow us to run your model, you must provide us two files, as mentioned
[here](../howto-submit.md):
 - a `requirements.txt` file that lists your dependencies; and
 - your modified `evaluator.py` script.
 
You must modify the script (see the function `generate_predictions`) in order to run your model and
write the results to the provided prediction file. In particular, you will receive two parameters:
 - `input_file_path`, which is the path to the file containing the data with the input.
 - `pred_file_path`, which is the path where you should write the model output.

Note that any modification to the ``evaluator.py`` script outside of this function will be ignored.
If this function is not implemented or if something breaks during evaluation, your model will not be ranked,
and you will be penalized.

### Testing your modified evaluation script

In order to test your evaluation script, run the following from within your submission directory:

    # create and source a clean virtual env
    pip install -e requirements.txt
    # run the evaluator
    python evaluator.py --input-file-path /project/cq-training-1/project2/data/train.lang1
                        --target-file-path /project/cq-training-1/project2/data/train.lang2

This will give you the BLEU score on the training set we provided. Make sure that all the requirements
are correctly loaded, the script runs and the result makes sense.