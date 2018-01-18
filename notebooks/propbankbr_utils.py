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
	
	P = max(df['P']) # gets the preposition
	Stest = min(df.loc[df['P']> P-testN,'S']) # from proposition gets the sentence	
	dftest= df[df['S']>=Stest]

	Svalid = min(df.loc[df['P']> P-(testN+validN),'S']) # from proposition gets the sentence	
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
	df_dep2= df_dep[['FUNC', 'DTREE', 'S', 'P' ]]
	# import code; code.interact(local=dict(globals(), **locals()))		
	usecols= ['ID', 'S', 'P',  'FORM', 'LEMMA', 'GPOS', 'MORF', 
		'DTREE', 'FUNC', 'CTREE', 'PRED',  'ARG0', 'ARG1', 'ARG2','ARG3', 'ARG4', 
		'ARG5', 'ARG6'
	]

	df= pd.concat((df_const, df_dep2), axis=1)
	df= df.applymap(trim)

	return df[usecols] 

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
	proposition_count=[]
	p_count=0
	M=max(mappings_inv.keys())   # max number of fields
	for line in open(filename):			
		end_of_sentence= (len(line)==1)
		if end_of_sentence: 
			s_count+=1

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
	
	sentences['S']= sentence_count # adds a new column with number of sentences
	sentences['P']= proposition_count # adds a new column with number of propositions
	df = pd.DataFrame.from_dict(sentences)				
	# garantee a friedlier ordering of the columns
	cols=['ID', 'S' , 'P'] + list(mappings.keys())[1:]
	df = df[cols]
	
	return df



def get_signature(mappings): 
	return {k:[] for k in mappings}


if __name__== '__main__':		
		
	df = propbankbr_parser()
	dfdevel, dfvalid, dftest=propbankbr_split(df)
	print(dfdevel.head())
	print(dfvalid.head())
	print(dftest.head())

	#Splits saves according to artur beltrao's dissertation
	# dfdevel.to_csv('../propbankbr/default_devel.csv')
	# dfvalid.to_csv('../propbankbr/default_valid.csv')
	# dftest.to_csv('../propbankbr/default_test.csv')

	#Splits saves according to zhou's article
	dfdevel.to_csv('../propbankbr/default_devel.csv')
	dfvalid.to_csv('../propbankbr/default_valid.csv')
	dftest.to_csv('../propbankbr/default_test.csv')



	# stats = propbankbr_argument_stats(df)
	# print(stats)







