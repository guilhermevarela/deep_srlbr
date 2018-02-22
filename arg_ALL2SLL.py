import numpy as np 
import pandas as pd 
import sys
sys.path.append('datasets/')
from data_propbankbr import propbankbr_lazyload

def propbankbr_transform_arg12arg0(propositions, arguments):
	isopen=False
	prev_tag=''
	prev_prop=-1
	new_tags=[]
	# propositions= df['P'].tolist()
	# tags= df['ARG_1'].tolist()
	for prop, tag in zip(propositions, arguments):
		if prop != prev_prop:
			prev_tag=''
			if isopen: # Close 
				new_tags[-1]+= ')' 
				isopen=False
			
		if tag != prev_tag:			
			if prev_tag != '*' and prev_prop == prop:
				new_tags[-1]+= ')' 
				isopen=False

			if tag != '*':	
				new_tag= '({:}*'.format(tag)
				isopen=True
			else:
				new_tag='*'
		elif prev_prop == prop:
			new_tag= '*'


		prev_tag= tag 	
		prev_prop= prop
		new_tags.append(new_tag)

	if isopen:
		new_tags[-1]+=')'		
		isopen=False
		
	return new_tags	
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


