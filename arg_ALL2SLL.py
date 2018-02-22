import numpy as np 
import pandas as pd 
import sys
sys.path.append('datasets/')
from data_propbankbr import propbankbr_lazyload


if __name__ == '__main__':
	df=propbankbr_lazyload('zhou_valid')	
	# propositions= [5148]*15 + [5149]*15 +[5150]*15
	# tags= ['AM-NEG','V','A2','A1','A1','A1','A1','A1','A1','A1','A1','A1','A1','A1','A1','*','*','*','V','A1','A1','A1','A1','A1','A1','A1','A1','A1','A1','A1','*','*','*','*','*','A1','A1','A1','V','A2','A2','A2','A2','A2','A2']
	# iters = list(np.arange(15))*3
	propositions= df['P'].tolist()
	arguments= df['ARG_1'].tolist()

	df['TEST']	= propbankbr_transform_arg12arg0(propositions, arguments)
	df.to_csv('test.csv')
	# for i,tag,ntag in zip(iters, tags,new_tags):
	# 	print('{:}\t{:}\t{:}\t'.format(i,tag,ntag))


