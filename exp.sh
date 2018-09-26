#!/bin/sh
#This script runs a series of benchmark experiments
python srl.py 16 16 --target R IOB --embs-model wrd50 --ru GRU --epochs 250
#python srl.py 32 16 --target R IOB --embs-model wan300 --epochs 250 --batch_size 100
#python srl.py 16 16 16 16 --target R IOB --embs-model wan100 --epochs 250 --batch_size 100
#python srl.py 16 16 --target R IOB --embs-model glo50 --epochs 250 --batch_size 100
#python srl.py 16 16 --target R IOB --embs-model wan50 --ru GRU --epochs 250 --batch_size 100
