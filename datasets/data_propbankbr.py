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
import os.path

PROPBANKBR_PATH='datasets/conll/'
TARGET_PATH='datasets/csvs/'

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
	'ID', 'S', 'P', 'P_S', 'FORM', 'LEMMA', 'PRED', 'FUNC', 'M_R', 'ARG_0', 'ARG_1'
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

def propbankbr_lazyload(dataset_name='zhou'):
	dataset_path= TARGET_PATH + '/{}.csv'.format(dataset_name)	
	if os.path.isfile(dataset_path):
		df= pd.read_csv(dataset_path)		
	else:
		df= propbankbr_parser2()
		propbankbr_persist(df, split=True, dataset_name=dataset_name)		  
	return df 
		
def propbankbr_persist(df, split=True, dataset_name='zhou'):
	df.to_csv('{}{}.csv'.format(TARGET_PATH ,dataset_name))
	if split: 
		dftrain, dfvalid, dftest=propbankbr_split(df)
		dftrain.to_csv( TARGET_PATH + '/{}_train.csv'.format(dataset_name))
		dfvalid.to_csv( TARGET_PATH + '/{}_valid.csv'.format(dataset_name))
		dftest.to_csv(  TARGET_PATH + '/{}_test.csv'.format(dataset_name))
	


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

	dftrain= df[df['S']<Svalid]
	return dftrain, dfvalid, dftest


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
	df_const= _const_read()
	df_dep= _dep_read() 
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

def propbankbr_parser2(ctx_p_size=1):		
	'''
	Parsers according to 
	End-to-end Learning of Semantic Role Labeling Using Recurrent Neural Networks
	Zhou and Xu, 2016

	'ID'  	: Contador de tokens que inicia em 1 para cada nova proposição
	'S'  		: Contador de sentencas
	'P'  		: Contador de predicatos
	'P_S'  	: Contador de predicatos por sentenças
	'FORM'  : Forma da palavra ou sinal de pontuação
	'LEMMA' : Lema gold-standard da FORM 
	'PRED'  : Predicatos semânticos na proposição repetido por todas as etiquetas
	'FUNC'  : Predicatos semânticos na proposição conforme notação propbank
	'CTX_P'  : Contexto do predicato ( tamanho de palavras ao redor do predicado)
						Ex:   
						FORM: ['Obras', 'foram', 'feitas', 'por', 'empreeiteiras'] 
						PRED: [		 '-',   'ser',      '-',   '-', '-']
						CTXP (TAM: 1): ['Obras foram feitas','Obras foram feitas','Obras foram feitas','Obras foram feitas','Obras foram feitas']
						CTXP (TAM: 2): ['. Obras foram feitas por','. Obras foram feitas por','. Obras foram feitas por','. Obras foram feitas Por','. Obras foram feitas por']

	'M_R'  : Marcacao do predicato
						0 se o predicato não modifica
						1 se o predicato modifica
						Ex:   
						FORM: ['Obras', 'foram', 'feitas', 'por', 'empreeiteiras'] 
						PRED: [		 '-',   'ser',      '-',   '-', '-']
						M_R:  [			 0,       1,        1,     1,   1]

	'ARG_0'  : Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank
	'ARG_1'  : Papel Semântico do regente do argumento na árvore de dependência, conforme notação PropBank

	updates: 2018-02-01
	updates: 2018-02-20: added FUNC, relabeled  ARG_Y to ARG_1
	updates: 2018-02-24: added CTX_P
	'''

	df_const= _const_read()
	df_dep= _dep_read() 
	# preprocess
	df_dep2= df_dep[['FUNC', 'DTREE', 'S', 'P', 'P_S' ]]
	usecols= ['ID', 'S', 'P', 'P_S', 'FORM', 'LEMMA', 'PRED',  'ARG0', 'ARG1', 'ARG2','ARG3', 'ARG4', 
		'ARG5', 'ARG6'
	]

	df= pd.concat((df_const, df_dep2), axis=1)
	df= df[usecols] 	
	df= df.applymap(trim)

	
	d_sentence_size =_get_dict_sentence_size(df)
	d_sentence_numpredicates=_get_dict_sentence_numpredicates(df)
	d_sentence_predicates =_get_dict_sentence_predicates(df)

	N=0
	for s,sentence_sz in d_sentence_size.items():
		N+= sentence_sz*d_sentence_numpredicates[s] # COMPUTES sum(|S_i|*|P_ij|)

	Nind= 5 
	Xind=np.zeros((N, Nind),dtype=np.int32)
	Yind=np.zeros((N, Nind),dtype=np.int32)	
	
	P=   np.zeros((N,1),dtype=np.int32)		
	M_R= np.zeros((N,1),dtype=np.int32)	
	P_S= np.zeros((N,1),dtype=np.int32)	
	PRED=[]
	FUNC=[]
	
	ctx_p_keys=[k for k in range(-ctx_p_size, ctx_p_size+1)
						if k != 0 ] # exclude 0	which is already offered					
	ctx_p_values=[[] for _ in ctx_p_keys]
	d_ctx_p=dict(zip(ctx_p_keys,ctx_p_values))

	#Iterate item by item adjusting PRED, CTX_P, M_R, ARG
	#resulting df will have number of rows=|S_i|X|P_ij|
	usecols= ZHOU_HEADER[:-2] + ['CTX_P{:+d}'.format(key) for key in ctx_p_keys] + ZHOU_HEADER[-2:]

	Y=[] 
	x_in =0
	x_out=0
	p=0
	for sentence_id,sentence_sz in d_sentence_size.items():
		sentence_numpredicates = d_sentence_numpredicates[sentence_id]
		sentence_df=df[df['S']==sentence_id]		
		for p_s in range(sentence_numpredicates):		
			func=['-']*sentence_sz
			x_data=np.arange(x_in, x_in+sentence_sz).reshape((sentence_sz,1))
			y_data=np.array([0,1,4,5,7+p_s]).reshape((1,Nind))			
			
			Xind[x_out:x_out+sentence_sz,:]= np.tile(x_data, (1,Nind))
			Yind[x_out:x_out+sentence_sz,:]= np.tile(y_data, (sentence_sz,1))
			
			PRED+=[d_sentence_predicates[sentence_id][p_s]]*sentence_sz			

			#FUNC will be PRED if ARG_0 == (V*)			
			ifunc=np.argmax((sentence_df['PRED'] == d_sentence_predicates[sentence_id][p_s]).values)			
			func[ifunc]=d_sentence_predicates[sentence_id][p_s]			
			FUNC+=func 

			#D_CTX_P
			for ctx_p in d_ctx_p:
				this_lemma=['-']
				if ifunc + ctx_p >= 0  and ifunc + ctx_p < sentence_sz: 
					this_lemma= [sentence_df.loc[:, ('LEMMA')].values[ifunc + ctx_p]]			
				d_ctx_p[ctx_p]+= this_lemma*sentence_sz
			
			
			ind=(sentence_df['P_S']>=p_s+1).as_matrix()
			M_R[x_out:x_out+sentence_sz,:]= (ind.reshape(sentence_sz,1)).astype(np.int32)
			P[x_out:x_out+sentence_sz,:]=p+1
			P_S[x_out:x_out+sentence_sz,:]=p_s
			x_out+=sentence_sz 
			p+=1
		x_in+=sentence_sz


	
	#CONVERT INDEX into DF
	data= df.as_matrix()[Xind,Yind]	
	zhou_df= 	pd.DataFrame(data=data, columns=['ID', 'S', 'FORM', 'LEMMA', 'ARG_0'])

	#NEW COLUMNS 
	zhou_df['FUNC']=FUNC
	zhou_df['PRED']=PRED
	zhou_df['M_R']=M_R
	zhou_df['P']=P
	zhou_df['P_S']=P_S
	
	for key, values in d_ctx_p.items():		
		zhou_df['CTX_P{:+d}'.format(key)]=values		
	#DATA TRANSORMATIONS: BY ITERATING ITEM BY ITEM
	zhou_df['ARG_1']= propbankbr_transform_arg02arg1(P, zhou_df['ARG_0'].tolist())
	zhou_df.index.name= 'IDX'

	# d_ctx_p= propbankbr_transform_ctx_p(zhou_df, size=1)
	return zhou_df[usecols]

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

def _const_read():
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
	# import code; code.interact(local=dict(globals(), **locals()))		
	df = pd.read_csv(filename, sep='\t', header=None, index_col=False, names=CONST_HEADER, dtype=str)
	
	del df['IGN1'] 
	del df['IGN2'] 
	del df['IGN3'] 

	return df 

def _dep_read():
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
	p_count=0                    # counter over the number of predicates
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


def propbankbr_transform_arg12arg0(propositions, arguments):
	isopen=False
	prev_tag=''
	prev_prop=-1
	new_tags=[]
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

def propbankbr_transform_arg02arg1(propositions, arguments):
	'''
		Converts default argument 0 into argument 1  format for easier softmax

	'''
	prev_tag=''
	prev_prop=-1
	new_tags=[]
	for prop, tag in zip(propositions, arguments):
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



def get_signature(mappings): 
	return {k:[] for k in mappings}

def _get_dict_sentence_size(df):
	xdf= df[['S','P']].pivot_table(index=['S'], aggfunc=len)
	d=dict(zip(xdf.index, xdf['P']))
	return d

def _get_dict_sentence_numpredicates(df):

	xdf= df[['S', 'P_S']].drop_duplicates(subset=['S', 'P_S'],keep='last',inplace=False)	
	d= xdf.pivot_table(index='S', values='P_S', aggfunc=max).T.to_dict('int')
	d=d['P_S']
	return d

def _get_dict_sentence_predicates(df):
	xdf= df[['S', 'PRED']]
	xdf= xdf[xdf['PRED'] != '-']	
	d= {}
	for seq, pred in zip(xdf['S'], xdf['PRED']):
		if seq in d:
			d[seq]+=[pred]
		else:
			d[seq]=[pred]
	return d

def trim(val):
	if isinstance(val, str):
		return val.strip()
	return val	


if __name__== '__main__':		
	print('Parsing propbank')
	df =propbankbr_parser2(ctx_p_size=3)
	
	print('Done. with shape=', df.shape)
	print('Spliting dataset')
	df_train, df_valid, df_test =propbankbr_split(df)
	print('Train. with shape=', df_train.shape)
	print('Valid. with shape=', df_valid.shape)
	print('Test.  with shape=', df_test.shape)
	print('Persisting propbank')
	propbankbr_persist(df)

	
	