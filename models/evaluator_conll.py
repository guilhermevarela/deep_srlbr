'''
	This module works a wrapper for pearl's conll task evaluation script

	Created on Mar 15, 2018
	
	@author: Varela


	ref:
	
	CONLL 2005 SHARED TASK
		HOME: http://www.lsi.upc.edu/~srlconll/
		SOFTWARE: http://www.lsi.upc.edu/~srlconll/soft.html

'''
import sys
import subprocess
import pandas as pd 
PEARL_SRLEVAL_PATH='./srlconll-1.1/bin/srl-eval.pl'
class EvaluatorConll(object):

	def __init__(self, target_dir):
		#save temporary files in root_path
		self.root_dir= '/'.join(sys.path[0].split('/')[:-1])
		if (target_dir):
			self.target_dir= target_dir

		self.f1= -1
		self.prec=-1
		self.rec=-1
		
	def evaluate(self, S, P, FUNC, T, Y):
		'''
			Evaluates the conll scripts returning total precision, recall and F1
				if self.target_dir is set will also save conll.txt@self.target_dir
			args:
				PRED		 	.: list<string> predicates according to PRED column
				T 				.: list<string> target according to ARG column
				Y 				.: list<string> 
			returns:
				prec			.: float<> precision
				rec       .: float<> recall 
				f1        .: float<> F1 score
		'''
		raise NotImplementedError('evaluate is not implemented')

			
	def _exec(self, S, P, FUNC, T, Y):	
		'''
			Performs a 6-step procedure in order to use the script evaluation
			1) Formats 		.: inputs in order to obtain proper conll format ()
			2) Saves      .:  two tmp files tmpgold.txt and tmpy.txt on self.root_dir.
			3) Run 			  .:  the perl script using subprocess module.
			4) Parses     .:  parses results from 3 in variables self.f1, self.prec, self.rec. 
			5) Stores     .:  stores results from 3 in self.target_dir (optional).
			6) Cleans     .:  files left from step 2.
		'''

		#Step 1 - 
		#Step 3 - Popen runs the pearl script storing in the variable PIPE
		pipe= subprocess.Popen(['perl',PEARL_SRLEVAL_PATH,ft.filename, fy.filename], stdout=subprocess.PIPE)

		out, err = pipe.communicate()
		if (err):
			raise Exception('pearl script failed with message:\n{:}'.format(err))


def outputs_conll(S, P, FUNC, T):
	'''
		Performs convertions from normalized format (1 PREDICATE at TIME) vs
	'''
	df= pd.DataFrame(data=[S,P,FUNC,T,Y], columns=['S','P','FUNC','T'])
	
	#Replace '-' making it easier to concatenate
	sub_fn= lambda x: re.sub('-', '', x)
	df['FUNC'] = df['FUNC'].apply(sub_fn)
    
	s0= min(df['S'])
	sn= max(df['S'])
	for i,si in enumerate(range(s0,sn+1)):
	    dfsi=df.loc[df['S'] == si,:]
	    p0= min(dfsi['P'])
	    pn= max(dfsi['P'])    
	    for j,p in enumerate(range(p0,pn+1)): # concatenate by argument columns
	        dfpj=dfsi.loc[df['P']==p,:].reset_index(drop=True)
	        if j==0:
	            dfp=dfpj[['FUNC',target_column]]
	            dfp=dfp.rename(columns={target_column: 'ARG0'})
	        else:
	            dfp['FUNC']=dfp['FUNC'].map(str).values + dfpj['FUNC'].map(str).values
	            dfp= pd.concat((dfp, dfpj[target_column]), axis=1)
	            dfp=dfp.rename(columns={target_column: 'ARG{:d}'.format(p-p0)})            
	    if i==0:        
	        dfconll= dfp
	    else:
	        dfconll= dfconll.append(dfp)
	    blank_series= pd.Series([None]*dfconll.shape[1], index=dfconll.columns)    
	    dfconll= dfconll.append(blank_series, ignore_index=True)
	        
	#Replace '-' making it easier to concatenate
	sub_fn= lambda x: '-' if isinstance(x,str) and len(x) == 0 else x
	dfconll['FUNC'] = dfconll['FUNC'].apply(sub_fn)
    
	#Replace Nans ( from skipping lines to empty)
	dfconll= dfconll.fillna('')
    
	#Order columns to fit conll standard
	num_columns=len(dfconll.columns)
	usecolumns=['FUNC'] + ['ARG{:d}'.format(i) for i in range(num_columns-1)]
	return dfconll[usecolumns]





