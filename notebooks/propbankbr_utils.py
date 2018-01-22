'''
	Author: Guilherme Varela

	Performs dataset build according to refs/1421593_2016_completo
	1. Merges PropBankBr_v1.1_Const.conll.txt and PropBankBr_v1.1_Dep.conll.txt as specified on 1421593_2016_completo
	2. Parses new merged dataset into train (development, validation) and test, so it can be benchmarked by conll scripts 

'''
import sys 
import random 
import pandas as pd 
import numpy as np 
import re

PROPBANKBR_PATH='../propbankbr/'

#MAPS the filename and output fields to be harvested
CONST_HEADER=[
	'ID','FORM','LEMMA','GPOS','MORF', 'IGN1', 'IGN2', 
	'CTREE','IGN3', 'PRED','ARG0','ARG1','ARG2','ARG3',
	'ARG4','ARG5','ARG6'
]
DEP_HEADER=[
	'ID','FORM','LEMMA','GPOS','MORF', 'DTREE', 'FUNC', 
	'IGN1', 'PRED','ARG0','ARG1','ARG2','ARG3',
	'ARG4','ARG5','ARG6'
]
ZHOU_HEADER=[
	'ID', 'S', 'P', 'FORM', 'LEMMA', 'PRED', 'M_R', 'LABEL'
]
MAPPER= {
	'CONST': { 
		'filename': 'PropBankBr_v1.1_Const.conll.txt',
		'mappings': {
			'ID':0,
			'FORM':1,
			'LEMMA':2,
			'GPOS':3,
			'MORF':4,
			'CTREE':7,
			'PRED':9,
			'ARG0':10,
			'ARG1':11,
			'ARG2':12,
			'ARG3':13,
			'ARG4':14,
			'ARG5':15,
			'ARG6':16,
		}
	}, 
	'DEP': { 
		'filename': 'PropBankBr_v1.1_Dep.conll.txt',
		'mappings': {
			'ID':0,
			'FORM':1,
			'LEMMA':2,
			'GPOS':3,
			'MORF':4,
			'DTREE':5,
			'FUNC':6,
			'PRED':8,
			'ARG0':9,
			'ARG1':10,
			'ARG2':11,
			'ARG3':12,
			'ARG4':13,
			'ARG5':14,
			'ARG6':15,
		}
	}
}

def trim(val):
	if isinstance(val, str):
		return val.strip()
	return val

def propbankbr_split(df, testN=263, validN=569):
	'''
		Splits propositions into test & validation following convetions set by refs/1421593_2016_completo
			|development data|= trainN + validationN 
			|test data|= testN

	'''	
	# import code; code.interact(local=dict(globals(), **locals()))		
	P = max(df['P_S']) # gets the preposition
	Stest = min(df.loc[df['P_S']> P-testN,'S']) # from proposition gets the sentence	
	dftest= df[df['S']>=Stest]

	Svalid = min(df.loc[df['P_S']> P-(testN+validN),'S']) # from proposition gets the sentence	
	dfvalid= df[((df['S']>=Svalid) & (df['S']<Stest))]

	dfdevel= df[df['S']<Svalid]
	return dfdevel, dfvalid, dftest

def propbankbr_argument_stats(df):
	'''
		Removes '', NaN, '*', 'V' in order to account only for valid tags
	'''
	def argument_transform(val):
		if isinstance(val, str):
			val = re.sub(r'C\-|\(|\)|\*|\\n| +', '',val)			
			val = re.sub(r' ', '',val)			
		else: 
			val=''
		return val

	dfarg= df[['ARG0','ARG1','ARG2','ARG3','ARG4','ARG5','ARG6']]
	dfarg= dfarg.applymap(argument_transform)
	values= dfarg.values.flatten()
	values= list(filter(lambda a : a != '' and a != 'V', values))

	stats= {k:0 for k in set(values)}
	for val in list(values): 
		stats[val]+=1
	return stats

def propbankbr_parser():		
	'''
	Uses refs/1421593_2016_completo.pdf

	'ID'  	: Contador de tokens que inicia em 1 para cada nova proposição
	'FORM'  : Forma da palavra ou sinal de pontuação
	'LEMMA' : Lema gold-standard da FORM 
	'GPOS'  : Etiqueta part-of-speech gold-standard
	'MORF'  : Atributos morfológicos  gold-standard
	'DTREE' : Árvore Sintagmática gold-standard completa	
	'FUNC'  : Função Sintática do token gold-standard para com seu regente na árvore de dependência
	'CTREE' : Árvore Sintagmática gold-standard completa
	'PRED'  : Predicatos semânticos na proposição
	'ARG0'  : 1o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'ARG1'  : 2o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'ARG2'  : 3o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'ARG3'  : 4o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'ARG4'  : 5o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'ARG5'  : 6o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'ARG6'  : 7o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'''
	df_const= propbankbr_const_read()
	df_dep= propbankbr_dep_read() 
	# preprocess
	df_dep2= df_dep[['FUNC', 'DTREE', 'S', 'P', 'P_S' ]]
	usecols= ['ID', 'S', 'P', 'P_S',  'FORM', 'LEMMA', 'GPOS', 'MORF', 
		'DTREE', 'FUNC', 'CTREE', 'PRED',  'ARG0', 'ARG1', 'ARG2','ARG3', 'ARG4', 
		'ARG5', 'ARG6'
	]

	df= pd.concat((df_const, df_dep2), axis=1)
	df= df[usecols] 
	df= df.applymap(trim)

	return df

def propbankbr_parser2():		
	'''
	Parsers according to 
	End-to-end Learning of Semantic Role Labeling Using Recurrent Neural Networks
	Zhou and Xu, 2016

	'ID'  	: Contador de tokens que inicia em 1 para cada nova proposição
	'S'  		: Contador de sentencas
	'P'  		: Contador de predicatos
	'FORM'  : Forma da palavra ou sinal de pontuação
	'LEMMA' : Lema gold-standard da FORM 
	'PRED'  : Predicatos semânticos na proposição
	'CTX_P'  : Contexto do predicato 
						Ex:   
						FORM: ['Obras', 'foram', 'feitas', 'por', 'empreeiteiras'] 
						PRED: [		 '-',   'ser',      '-',   '-', '-']
						CTXP: ['foram feitas','foram feitas','foram feitas','foram feitas','foram feitas']

	'M_R'  : Marcacao do predicato
						0 se o predicato não modifica
						1 se o predicato modifica
						Ex:   
						FORM: ['Obras', 'foram', 'feitas', 'por', 'empreeiteiras'] 
						PRED: [		 '-',   'ser',      '-',   '-', '-']
						M_R:  [			 0,       1,        1,     1,   1]

	'ARG'  : Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'''

	df_const= propbankbr_const_read()
	df_dep= propbankbr_dep_read() 
	# preprocess
	df_dep2= df_dep[['FUNC', 'DTREE', 'S', 'P', 'P_S' ]]
	usecols= ['ID', 'S', 'P', 'P_S',  'FORM', 'LEMMA', 'PRED',  'ARG0', 'ARG1', 'ARG2','ARG3', 'ARG4', 
		'ARG5', 'ARG6'
	]

	df= pd.concat((df_const, df_dep2), axis=1)
	df= df[usecols] 	
	df= df.applymap(trim)

	#Iterate item by item adjusting PRED, CTX_P, M_R, ARG
	#resulting df will have number of rows=|S_i|X|P_ij|
	usecols= ZHOU_HEADER

	
	slen_dict =get_sentence_size(df)
	pslen_dict=get_number_predicates_in_sentence(df)
	pred_dict =get_nominal_predicates_in_sentence(df)
	

	N=0
	for s,l in slen_dict.items():
		N+= l*pslen_dict[s] # COMPUTES sum(|S_i|*|P_ij|)

	
	Xind=np.zeros((N,5),dtype=np.int32)
	Yind=np.zeros((N,5),dtype=np.int32)	
	
	P=   np.zeros((N,1),dtype=np.int32)		
	M_R= np.zeros((N,1),dtype=np.int32)	
	PRED=[]
	x_in =0
	x_out=0
	for s,l in slen_dict.items():
		n_p = pslen_dict[s]
		sdf=df[df['S']==s]
		for p in range(n_p):
			x_data=np.arange(x_in, x_in+l).reshape((l,1))
			y_data=np.array([0,1,4,5,7+p]).reshape((1,5))			
			
			Xind[x_out:x_out+l,:]= np.tile(x_data, (1,5))
			Yind[x_out:x_out+l,:]= np.tile(y_data, (l,1))
			
			PRED+=[pred_dict[s][p]]*l			

			ind=(sdf['P_S']>=p+1).as_matrix()
			M_R[x_out:x_out+l,:]= (ind.reshape(l,1)).astype(np.int32)
			P[x_out:x_out+l,:]=p+1

			x_out+=l 
		x_in+=l 

	#CONVERT INDEX into DF
	data= df.as_matrix()[Xind,Yind]
	zhou_df= 	pd.DataFrame(data=data, columns=['ID', 'S', 'P_S','FORM', 'LEMMA', 'LABEL'])
	zhou_df['PRED']=PRED
	zhou_df['M_R']=M_R
	zhou_df['P']=P
	zhou_df= zhou_df[usecols]

	return zhou_df

def propbankbr_const_read():
	'''
		Reads the file 'PropBankBr_v1.1_Const.conll.txt' returning a pandas DataFrame
		the format is as follows
		'ID'  	: Contador de tokens que inicia em 1 para cada nova proposição
		'FORM'  : Forma da palavra ou sinal de pontuação
		'LEMMA' : Lema gold-standard da FORM 
		'GPOS'  : Etiqueta part-of-speech gold-standard
		'MORF'  : Atributos morfológicos  gold-standard
		'CTREE' : Árvore Sintagmática gold-standard completa
		'PRED'  : Predicatos semânticos na proposição
		'ARG0'  : 1o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG1'  : 2o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG2'  : 3o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG3'  : 4o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG4'  : 5o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG5'  : 6o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG6'  : 7o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'''

	
	filename= PROPBANKBR_PATH + 'PropBankBr_v1.1_Const.conll.txt'

	df = pd.read_csv(filename, sep='\t', header=None, index_col=False, names=CONST_HEADER, dtype=str) 	
	
	del df['IGN1'] 
	del df['IGN2'] 
	del df['IGN3'] 

	return df 

def propbankbr_dep_read():
	'''
		Reads the file 'PropBankBr_v1.1_Dep.conll.txt' returning a pandas DataFrame
		the format is as follows, 
		'ID'  	: Contador de tokens que inicia em 1 para cada nova proposição
		'FORM'  : Forma da palavra ou sinal de pontuação
		'LEMMA' : Lema gold-standard da FORM 
		'GPOS'  : Etiqueta part-of-speech gold-standard
		'MORF'  : Atributos morfológicos  gold-standard
		'DTREE' : Árvore Sintagmática gold-standard completa
		'FUNC'  : Função Sintática do token gold-standard para com seu regente na árvore de dependência
		'PRED'  : Predicatos semânticos na proposição
		'ARG0'  : 1o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG1'  : 2o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG2'  : 3o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG3'  : 4o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG4'  : 5o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG5'  : 6o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
		'ARG6'  : 7o Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'''
	#malformed file prevents effective use of pd.read_csv
	filename=PROPBANKBR_PATH + MAPPER['DEP']['filename']
	mappings=MAPPER['DEP']['mappings']
	mappings_inv= {v:k for k,v in mappings.items()}
	sentences=get_signature(mappings)
	sentence_count=[]
	s_count=1 									 # counter over the number of sentences
	p_count=1                    # counter over the number of predicates
	ps_count=0									 # predicate per sentence
	proposition_count=[]	
	proposition_per_sentence_count=[]	

	M=max(mappings_inv.keys())   # max number of fields

	for line in open(filename):			
		end_of_sentence= (len(line)==1)
		if end_of_sentence: 
			s_count+=1
			ps_count=0


		if not(end_of_sentence):
			data= line.split(' ')					
			data= filter(lambda s: s != '' and s !='\n', data)			
			values= list(data)

			key_max=0
			for keys_count, val in enumerate(values): 
				if keys_count in mappings_inv.keys():
					key=mappings_inv[keys_count]	
					
					if (key=='PRED'):
						if (val != '-'):
							p_count+=1
							ps_count+=1	

					if (key[:3]=='ARG'):						
						val= val.translate(str.maketrans('','','\n()*')) 
						sentences[key].append(val)
					else:
						sentences[key].append(val)
				key_max=keys_count

			#	fills remaning absent/optional ARG arrays with None
			# garantees all arrays have the same number of elements
			for keys_count in range(key_max+1, M+1, 1): 
				key=mappings_inv[keys_count]	
				sentences[key].append(None)
			
			sentence_count.append(s_count)
			proposition_count.append(p_count)
			proposition_per_sentence_count.append(ps_count)
	
	sentences['S']= sentence_count # adds a new column with number of sentences
	sentences['P']= proposition_count # adds a new column with number of propositions
	sentences['P_S']=proposition_per_sentence_count # adds a new column with number of propositions per sentence

	df = pd.DataFrame.from_dict(sentences)				
	# garantee a friedlier ordering of the columns
	cols=['ID', 'S' , 'P', 'P_S'] + list(mappings.keys())[1:]
	df = df[cols]
	
	return df



def get_signature(mappings): 
	return {k:[] for k in mappings}

def get_sentence_size(df):
	xdf= df[['S','P']].pivot_table(index=['S'], aggfunc=len)
	x=dict(zip(xdf.index, xdf['P']))
	return x

def get_number_predicates_in_sentence(df):
	xdf= df[['S', 'P_S']].drop_duplicates(subset=['S', 'P_S'],keep='last',inplace=False)
	x_dict= xdf.set_index('S').T.to_dict('int')
	x_dict=x_dict['P_S']
	return x_dict

def get_nominal_predicates_in_sentence(df):
	xdf= df[['S', 'PRED']]
	xdf= xdf[xdf['PRED'] != '-']	
	d= {}
	for seq, pred in zip(xdf['S'], xdf['PRED']):
		if seq in d:
			d[seq]+=[pred]
		else:
			d[seq]=[pred]

	return d

if __name__== '__main__':		
		
	df = propbankbr_parser2()
	# import code; code.interact(local=dict(globals(), **locals()))		
	dfdevel, dfvalid, dftest=propbankbr_split(df)
	# print(dfdevel.head())
	# print(dfvalid.head())
	# print(dftest.head())

	#Splits saves according to artur beltrao's dissertation
	dfdevel.to_csv('../propbankbr/zhou_devel.csv')
	dfvalid.to_csv('../propbankbr/zhou_valid.csv')
	dftest.to_csv('../propbankbr/zhou_test.csv')

	df.to_csv('../propbankbr/zhou.csv')
	










