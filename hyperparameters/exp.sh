#!/bin/sh
#This script runs a series of benchmark experiments
python srl.py 16 16 --target R IOB --embs-model wan50 --epochs 250 --ru LSTM
python srl.py 16 16 16 16 --target R IOB --embs-model wan100 --epochs 250 --ru LSTM
python srl.py 16 16 --target R IOB --embs-model wan100 --epochs 250 --ru LSTM
python srl.py 32 16 --target R IOB --embs-model wan300 --epochs 250 --ru LSTM
python srl.py 16 16 --target R IOB --embs-model wrd50 --epochs 250 --ru LSTM
python srl.py 16 16 --target R IOB --embs-model glo50 --epochs 250 --ru LSTM

python srl.py 16 16 --target R IOB --embs-model wan50 --epochs 250 --r-depth 1
python srl.py 16 16 --target R IOB --embs-model wan50 --epochs 250 --r-depth 2

python srl.py 16 16 --target R IOB --embs-model wan100 --epochs 250 --r-depth 1
python srl.py 16 16 --target R IOB --embs-model wan100 --epochs 250 --r-depth 2

python srl.py 16 16 16 --target R IOB --embs-model wan100 --epochs 250 --r-depth 1
python srl.py 16 16 16 --target R IOB --embs-model wan100 --epochs 250 --r-depth 2
python srl.py 16 16 16 --target R IOB --embs-model wan100 --epochs 250 --r-depth 3

python srl.py 16 16 --target R IOB --embs-model wan300 --epochs 250 --r-depth 1
python srl.py 16 16 --target R IOB --embs-model wan300 --epochs 250 --r-depth 2

python srl.py 16 16 --target R IOB --embs-model wrd50 --epochs 250 --r-depth 1
python srl.py 16 16 --target R IOB --embs-model wrd50 --epochs 250 --r-depth 2

python srl.py 16 16 --target R IOB --embs-model glo50 --epochs 250 --r-depth 1
python srl.py 16 16 --target R IOB --embs-model glo50 --epochs 250 --r-depth 2

