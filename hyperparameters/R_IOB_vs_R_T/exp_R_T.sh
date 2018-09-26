#!/bin/sh
#This script runs a series of benchmark experiments
python srl.py 16 16 --target R T --embs-model wrd50 --ru GRU --epochs 250
python srl.py 32 16 --target R T --embs-model wan300 --epochs 250 --batch_size 100
python srl.py 16 16 16 16 --target R T --embs-model wan100 --epochs 250 --batch_size 100
python srl.py 16 16 --target R T --embs-model glo50 --epochs 250 --batch_size 100
python srl.py 16 16 --target R T --embs-model wan50 --ru GRU --epochs 250 --batch_size 100

# outputs/1.0/wrd50/hs_16x16/ctxp_1/R_T/batch/lr_5.00e-03/2018-09-26 010721/
# outputs/1.0/wan300/hs_32x16/ctxp_1/R_T/batch/lr_5.00e-03/2018-09-26 022452/
# outputs/1.0/wan100/hs_16x16x16x16/ctxp_1/R_T/batch/lr_5.00e-03/2018-09-26 042931/
# outputs/1.0/glo50/hs_16x16/ctxp_1/R_T/batch/lr_5.00e-03/2018-09-26 042939/
# outputs/1.0/wan50/hs_16x16/ctxp_1/R_T/batch/lr_5.00e-03/2018-09-26 063434/