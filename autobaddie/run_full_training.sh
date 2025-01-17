#!/bin/bash

WORKDIR_BASE = # path to AutoBADDIE directory (ending in [...]/AutoBADDIE)
JOB_NAME='autoBADDIE_example' # desired name for chemical system, ex. "autoBADDIE_example"
DATE='TEST'

TRAINING_DATA_PATH=${WORKDIR_BASE}/training_data/class1 # path to AutoBADDIE directory (ending in [...]/AutoBADDIE)
TRAIN_AUTOPATH=${WORKDIR_BASE}/training_results  # path to [...]/AutoBADDIE/training_results directory
AUTOBADDIE_BASE=${WORKDIR_BASE}/autobaddie # path to [...]/AutoBADDIE/autobaddie directory
RAND=$RANDOM
echo beginning $RAND

mkdir -p output

python master_makedataset_argparse.py --job_name $JOB_NAME --date $DATE \
     --training_data_path $TRAINING_DATA_PATH \
     --train_autopath $TRAIN_AUTOPATH \
     --param_json base_job_details_opls --E_hyp 0 --dih_reg_hyp 0.1 --top_reg_hyp 0 \
     --autobaddie_base ${AUTOBADDIE_BASE} | tee ./output/${RAND}.out

#parse the conditionname from the output of making the dataset
CONDITION="$(grep condition: ./output/${RAND}.out | cut -d " " -f 2)"
JOBNAME="$(grep job ./output/${RAND}.out | cut -d " " -f 3)"

echo condition from ${RAND}.out is: $CONDITION
echo selfcontainedbase is ${AUTOBADDIE_BASE}
./master_train_argparse.sh -j ${JOBNAME} -c ${CONDITION} -r ${RAND} -b ${AUTOBADDIE_BASE} 
