import re 
import numpy as np 
import pandas as pd 
import sys
sys.path.append('datasets/')
from data_propbankbr import propbankbr_lazyload

def propbankbr_transform_arg02arg1(propositions, arguments):
	prev_tag=''
	prev_prop=-1
	new_tags=[]
	for prop, tag in zip(propositions, tags):
		if prev_prop == prop: 
			if (tag in ['*']): # Either repeat or nothing
				if (')' in prev_tag): 
					new_tag='*'
				else:	
					new_tag=new_tags[-1] # repeat last
			else:
				if  (')' in prev_tag): #last tag is closed, refresh					 
					new_tag=re.sub(r'\(|\)|\*|','',tag)					
				else:
					if prev_tag != tag and tag != '*)':
						new_tag=re.sub(r'\(|\)|\*|','',tag)					
					else:
						new_tag=new_tags[-1]			
		else: 
			if (tag in ['*']):
				new_tag='*'
			else:
				new_tag=re.sub(r'\(|\)|\*|','',tag)

		prev_tag= tag 	
		prev_prop= prop		
		new_tags.append(new_tag)
	return new_tags

if __name__ == '__main__':	
	df=propbankbr_lazyload('zhou_valid')	
	propositions= df['P'].tolist()
	tags= df['ARG_0'].tolist()


	df['TEST']	= propbankbr_transform_arg02arg1(propositions, tags)
	df.to_csv('test2.csv')



