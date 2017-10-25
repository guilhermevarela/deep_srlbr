'''
	Author: Guilherme Varela

	Performs dataset build according to refs/1421593_2016_completo
	1. Merges PropBankBr_v1.1_Const.conll.txt and PropBankBr_v1.1_Dep.conll.txt as specified on 1421593_2016_completo
	2. Parses new merged dataset into train (development, validation) and test, so it can be benchmarked by conll scripts 

'''


def propbankbr_parse_1():
	'''
		propbankbr_parse_1 : generates development, validation and test sets according to 1421593_2016_completo
	'''

	# How to represent the dataset?
	# which datastructure best represents the dataset?
	# An array of dictionaries
	# 	each example is a dictionary
	
	S= [] 
	signature= [
	('ID', []), ('FORM',[]), ('LEMMA',[]),('GPOS',[]),
	('MORF',[]), ('DTREE',[]),('FUNC',[]),('CTREE',[]), 
	('PRED',[]),('ARG', [])
	]
	
	i=0
	j=0 
	k=1
	p=1 
	r=0
	sequence=dict(signature)
	predicates={}
	arguments={}
	for line in open('PropBankBr_v1.1_Const.conll.txt'):		
		# every new sequence except the first is followed by
		if ((i == 0) | (line == '\n')):
			j=0 
			k+=1
			p=len(predicates)+1
			if len(sequence)>0: # excludes first
				S.append(sequence)
			sequence= dict(signature)

		if len(line)>1:
			seq= line.split('\t')
			seq= list(map(lambda s: s.replace(' ',''), seq))
			sequence['ID'].append(int(seq[0])
			sequence['FORM'].append(seq[1])
			sequence['LEMMA'].append(seq[2])
			sequence['GPOS'].append(seq[3])
			sequence['MORF'].append(seq[4])
			sequence['DTREE'].append(seq[5])
			sequence['FUNC'].append(seq[6])
			sequence['CTREE'].append(seq[7])
			sequence['PRED'].append(seq[8])
			# last segment is a function over the number of predicates
			args=seq[9:-1]
			for a, arg in enumerate(args):	
				#updates predicates
				if not((p+a) in predicates):
					predicates[p+a]=k 		
				#updates arguments	
				if arg != '*':
					r+=1
					arguments[r]=(i,j)
				sequence['ARG'].append(arg)
				j+=1			
		i+=1
	print('total sentences',len(S))
	print('total propositions',len(predicates))
	print('total arguments',len(arguments))
	print(S[0]['ID']) 
if __name__== '__main__':		
	propbankbr_parse_1()
	





