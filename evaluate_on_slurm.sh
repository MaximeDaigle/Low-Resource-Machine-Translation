#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

module --ignore-cache load python/3.7
source /project/cq-training-1/project2/submissions/team07/low_env/bin/activate

python --version
which python
which pip
pip freeze

echo ""
echo "Calling python evaluation script."
# test1
stdbuf -oL python -u ./evaluator.py --target-file-path /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/sub_test.lang2 --input-file-path /project/cq-training-1/project2/teams/team07/vecmap_fr_cased_punc/sub_test.lang1
