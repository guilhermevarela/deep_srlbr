#!/bin/sh
#This script runs a series of benchmark experiments
python srl.py 32 32 --target IOB --embs-model glo50 --epochs 30
python srl.py 64 64 --target IOB --embs-model glo50 --epochs 30

python srl.py 32 32 --target IOB --embs-model wan50 --epochs 30
python srl.py 64 64 --target IOB --embs-model wan50 --epochs 30

python srl.py 32 32 --target IOB --embs-model wan100 --epochs 30
python srl.py 64 64 --target IOB --embs-model wan100 --epochs 30

python srl.py 32 32 --target IOB --embs-model wan300 --epochs 30
python srl.py 64 64 --target IOB --embs-model wan300 --epochs 30

python srl.py 32 32 --target IOB --embs-model wrd50  --epochs 30
python srl.py 64 64 --target IOB --embs-model wrd50  --epochs 30
