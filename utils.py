'''
Created on Jan 24, 2018
	@author: Varela
	
	Auxiliary functions 
'''
import numpy as np 

def onehot_encode(strarray):
	'''
		Converts an array into a matrix represe
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