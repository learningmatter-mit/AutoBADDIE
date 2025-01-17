#!/bin/bash

source ~/.bashrc

AUTOBADDIE_BASE=/home/pleon/projects/AutoBADDIE/autobaddie
RAND=$RANDOM
echo beginning $RAND

mkdir -p output

python master_makedataset_argparse.py --job_name autoBADDIE_example --date TESTTESTTEST \
     --param_json base_job_details_opls --E_hyp 0 --dih_reg_hyp 0.1 --top_reg_hyp 0 \
     --training_data_path "/home/pleon/projects/AutoBADDIE/training_data/class1" \
     --train_autopath "/home/pleon/projects/AutoBADDIE/training_results" \
     --autobaddie_base ${AUTOBADDIE_BASE} | tee ./output/${RAND}.out

#parse the conditionname from the output of making the dataset
CONDITION="$(grep condition: ./output/${RAND}.out | cut -d " " -f 2)"
JOBNAME="$(grep job ./output/${RAND}.out | cut -d " " -f 3)"

echo condition from ${RAND}.out is: $CONDITION
echo selfcontainedbase is ${AUTOBADDIE_BASE}
./master_train_argparse.sh -j ${JOBNAME} -c ${CONDITION} -r ${RAND} -b ${AUTOBADDIE_BASE} 
