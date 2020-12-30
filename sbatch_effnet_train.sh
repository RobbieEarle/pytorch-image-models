#!/bin/bash
#SBATCH -p t4v2
#SBATCH --exclude=gpu102
#SBATCH --exclude=gpu115
#SBATCH --gres=gpu:1                        # request GPU(s)
#SBATCH --qos=normal
#SBATCH -c 4                                # number of CPU cores
#SBATCH --mem=8G                            # memory per node
#SBATCH --time=64:00:00                     # max walltime, hh:mm:ss
#SBATCH --array=0%1                    # array value
#SBATCH --output=logs_new/p1_eff_baseline_tl/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=p1_eff_baseline_tl

source ~/.bashrc
source activate ~/venvs/efficientnet_train

SAVE_PATH="$1"
SEED="$SLURM_ARRAY_TASK_ID"

touch /checkpoint/robearle/${SLURM_JOB_ID}
CHECK_PATH=/checkpoint/robearle/${SLURM_JOB_ID}

# Debugging outputs
pwd
which conda
python --version
pip freeze

echo ""
python -c "import torch; print('torch version = {}'.format(torch.__version__))"
python -c "import torch.cuda; print('cuda = {}'.format(torch.cuda.is_available()))"
python -c "import scipy; print('scipy version = {}'.format(scipy.__version__))"
python -c "import sklearn; print('sklearn version = {}'.format(sklearn.__version__))"
python -c "import matplotlib; print('matplotlib version = {}'.format(matplotlib.__version__))"
python -c "import tensorflow; print('tensorflow version = {}'.format(tensorflow.__version__))"
echo ""

echo "SAVE_PATH=$SAVE_PATH"
echo "SEED=$SEED"

python validate.py /scratch/ssd001/datasets/imagenet/val/ --model tf_efficientnet_b0 --pretrained --tf-preprocessing --tl --output $SAVE_PATH --resume $CHECK_PATH