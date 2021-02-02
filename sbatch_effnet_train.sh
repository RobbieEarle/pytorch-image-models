#!/bin/bash
#SBATCH -p t4v2
#SBATCH --exclude=gpu102
#SBATCH --exclude=gpu115
#SBATCH --gres=gpu:1                        # request GPU(s)
#SBATCH --qos=high
#SBATCH -c 32                                # number of CPU cores
#SBATCH --mem=128G                           # memory per node
#SBATCH --time=500:00:00                     # max walltime, hh:mm:ss
#SBATCH --array=42%1                    # array value
#SBATCH --output=logs_new/test_tl/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=test_tl

source ~/.bashrc
source activate ~/venvs/efficientnet_train

ACTFUN="$1"
LR="$2"
SEED="$SLURM_ARRAY_TASK_ID"

SAVE_PATH=~/pytorch-image-models/outputs/test_tl
CHECK_PATH="/checkpoint/$USER/${SLURM_JOB_ID}"
IMGNET_PATH=/scratch/ssd001/datasets/imagenet/

mkdir -p "$SAVE_PATH"
touch $CHECK_PATH

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

#~/utilities/log_gpu_cpu_stats 2000 0.5 -n -1 "${SAVE_PATH}/${SLURM_ARRAY_TASK_ID}_${SLURM_NODEID}_${SLURM_ARRAY_JOB_ID}_compute_usage.log"&
#export LOGGER_PID="$!"

python train2.py data caltech101 --model efficientnet_b0 -b 20 --actfun $ACTFUN --output $SAVE_PATH --check-path $CHECK_PATH --seed $SEED --epochs 450 --weight-init orthogonal --lr $LR --num-classes 101
#kill $LOGGER_PID