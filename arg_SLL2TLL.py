import re 
import numpy as np 
import pandas as pd 
import sys
sys.path.append('datasets/')
from data_propbankbr import propbankbr_lazyload

if __name__ == '__main__':	
	df=propbankbr_lazyload('zhou_valid')	
	propositions= df['P'].tolist()
	tags= df['ARG_0'].tolist()


	df['TEST']	= propbankbr_transform_arg02arg1(propositions, tags)
	df.to_csv('test2.csv')



