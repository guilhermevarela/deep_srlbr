'''
Created on Jan 24, 2018
	@author: Varela
	
	Auxiliary functions 
'''
import numpy as np 
import pandas as pd 

def onehot_encode(strarray):
	'''
		Converts an array into a matrix representation
		IN 
			strarray {str<N>}: 

		OUT:
			Yoh {int<N,M>}:

	'''
	
	keys=np.unique(strarray)
	N= len(strarray)
	M= len(keys)
	Yoh= np.zeros((N,M), dtype=np.int32)
	for n in range(N):
		Yoh[n,np.argwhere(strarray[n] == keys)] =1 

	return Yoh 


def shuffle_by_proposition(df, col_proposition='P_S', batch_sz=0):	
	'''
		Shuffles df by proposition column
		IN 
			df: pd.Dataframe<'ID', 'S', 'P', 'P_S', 'FORM', 'LEMMA', 'PRED', 'M_R', 'LABEL'>

		OUT:
			index 

			index_batch
	'''
		n_proposition= max(df[col_proposition])
		i_proposition= np.random.permutation(n_proposition)

		index=[]
		index_batch=[]
		if batch_sz>0:
			index_batch.append(0)

		for p in i_propositions:
			index+=sorted(df[df[col_proposition] == p].index)
			if batch_sz>0 & i % batch_sz==0:
				index_batch.append(len(index))
						
		index=np.array(index, dtype=np.int32)				
		if nargout==1:
			return index
		else:
			index_batch=np.array(index, dtype=np.int32)				
			return index, index_batch



def shuffle_by_rows(mtrx, row_index):
	'''
		Shuffles df by proposition column
		IN 
			mtrx int<N,M>

		OUT:
			df_shuffle			
	'''		
		N,M = mtrx.shape
		row_index= np.tile(row_index.reshape(N,1), (1,M))
		col_index= np.tile(np.arange(M).reshape(1,M), (N,1))
		return mtrx[row_index, col_index]

def token2vec(token, w2v):		
	try: 
		vec=w2v[token] 
	except:
		vec=w2v['unk'] 	
	return vec		




