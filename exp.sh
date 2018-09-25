#!/bin/sh
#This script runs a series of benchmark experiments
python srl.py 16 16 --target R IOB --embs-model wrd50 --ru GRU --epochs 500
python srl.py 16 16 --target R T --embs-model wan300 --epochs 500
python srl.py 16 16 16 16 --target R IOB --embs-model wan100 --epochs 500
python srl.py 16 16 --target R T --embs-model glo50 --epochs 500
python srl.py 16 16 --target R T --embs-model wan50 --ru GRU --epochs 500
