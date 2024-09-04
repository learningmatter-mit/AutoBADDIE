#!/bin/bash

source ~/.bashrc

SELF_CONTAINED_BASE=/home/pleon/projects/AuTopologyPipeline_detached/self_contained
RAND=$RANDOM
echo beginning $RAND

mkdir -p output

#should retry with: 221011_reg1_cutoff1e-3_a100_m100_i0_LR_01
python master_makedataset_argparse.py --job_name 240904_opls_test --date TESTTESTTEST2 \
     --param_json base_job_details_opls --E_hyp 0 --dih_reg_hyp 0.1 --top_reg_hyp 0 \
     --self_contained_base ${SELF_CONTAINED_BASE} | tee ./output/testfull_${RAND}.out


#parse the conditionname from the output of making the dataset
CONDITION="$(grep condition: ./output/testfull_${RAND}.out | cut -d " " -f 2)"
JOBNAME="$(grep job ./output/testfull_${RAND}.out | cut -d " " -f 3)"

echo condition from testfull_${RAND}.out is: $CONDITION
echo selfcontainedbase is ${SELF_CONTAINED_BASE}
./master_train_argparse.sh -j ${JOBNAME} -c ${CONDITION} -r ${RAND} -b ${SELF_CONTAINED_BASE} 
