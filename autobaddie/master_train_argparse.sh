#!/bin/bash
#SBATCH -n 20
#SBATCH -N 1
#SBATCH --gres=gpu:volta:1
#SBATCH -t 4000
#SBATCH --constraint=xeon-g6
#SBATCH --mem-per-cpu=5000
#SBATCH --no-requeue
#SBATCH --signal=B:2@300

source ~/.bashrc

while getopts j:c:r:b: flag
do
    case "${flag}" in
        j) JOBNAME=${OPTARG};;
        c) CONDITION=${OPTARG};;
        r) RANDOM_NUM=${OPTARG};;
        b) AUTOBADDIE_BASE=${OPTARG};;
    esac
done

echo the random number is $RANDOM_NUM and the jobname is $JOBNAME and the condition is $CONDITION and the base is $AUTOBADDIE_BASE
mv ./output/${RANDOM_NUM}.out ../train/${JOBNAME}/${CONDITION}/train_${CONDITION}.out

#concentration needs to include the repetition (ex: 1M_1, or 1M_3)
python master_train_argparse.py --job_name ${JOBNAME} --condition ${CONDITION} \
      --autobaddie_base $AUTOBADDIE_BASE
    #   --autobaddie_base $AUTOBADDIE_BASE >> ../train/${JOBNAME}/${CONDITION}/train_${CONDITION}.out
