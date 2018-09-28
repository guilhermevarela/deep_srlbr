#!/bin/sh
#This script runs a series of benchmark experiments
python srl.py 16 16 --target R IOB --embs-model wan50
python srl.py 16 16 16 16 --target R IOB --embs-model wan100
python srl.py 32 16 --target R IOB --embs-model wan300
python srl.py 16 16 --target R IOB --embs-model wrd50
python srl.py 16 16 --target R IOB --embs-model glo50

# R IOB vs R T 
#outputs/1.0/wrd50/hs_16x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-26 110634/
#outputs/1.0/wan300/hs_32x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-25 215108/
#outputs/1.0/wan100/hs_16x16x16x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-25 233958/
#outputs/1.0/glo50/hs_16x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-26 015154/
#outputs/1.0/wan50/hs_16x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-26 032941/

#LSTM vs GRU --> compare this files to the file above  
#outputs/1.0/glo50/hs_16x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-26 163929/
#outputs/1.0/wan100/hs_16x16x16x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-26 150942/
#outputs/1.0/wan300/hs_32x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-26 135958/
#outputs/1.0/wan50/hs_16x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-26 174440/
#outputs/1.0/wrd50/hs_16x16/ctxp_1/R_IOB/batch/lr_5.00e-03/2018-09-26 132312/
