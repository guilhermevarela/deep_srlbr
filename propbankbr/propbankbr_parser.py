'''
	Author: Guilherme Varela

	Performs dataset build according to refs/1421593_2016_completo
	1. Merges PropBankBr_v1.1_Const.conll.txt and PropBankBr_v1.1_Dep.conll.txt as specified on 1421593_2016_completo
	2. Parses new merged dataset into train (development, validation) and test, so it can be benchmarked by conll scripts 

'''
import sys 
import random 
# from string import maketrans
#MAPS the filename and output fields to be harvested
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
			'ARG7':17,				
		}
	}
}

def propbankbr_parser():
	'''
		propbankbr_parse_1 : generates development, validation and test sets according to 1421593_2016_completo
	'''

	# How to represent the dataset?
	# which datastructure best represents the dataset?
	# An array of dictionaries
	# 	each example is a dictionary
	filename= 	MAPPER['CONST']['filename']
	mappings=   MAPPER['CONST']['mappings']
	imappings= {v:k for k,v in mappings.items()}

	sentences= [] 	
	f_count=0
	s_count=-1 # sentence counter
	t_count=0  # tags counter
	p_count=0  # predicates count
	sentence=get_signature(mappings)
	predicates={}
	tags={}

	for line in open(filename):		
		# every new sentence except the first is followed by
		if ((f_count == 0) | (line == '\n')):			
			s_count+=1
			if f_count>0: # excludes first
				sentences.append(sentence)
			sentence= get_signature(mappings)

		if len(line)>1:
			# import code; code.interact(local=dict(globals(), **locals()))
			values= line.split('\t')
			values= list(map(lambda s: s.replace(' ',''), values))			
			for keys_count, val in enumerate(values): 
				if keys_count in imappings.keys() or (keys_count>10):
					if (keys_count<=10):
						key=imappings[keys_count]	

					if (keys_count>=10):						
						val= val.translate(str.maketrans('','','\n()*')) 
					sentence[key].append(val)
					
					if (key=='PRED'):
						if val != '-':
							predicates[p_count]=s_count 		
							p_count+=1

					if (key[:3] == 'ARG'): # SPECIAL CASE WITH MULTIPLE ARGS										
						#updates arguments	
						if val != '':
							tags[t_count]={ 
								'pos':(f_count,s_count,p_count),				
								'tags': val
							}
							t_count+=1
		f_count+=1		
		sys.stdout.write('#rows:%05d\t#sentences:%05d\t#predicates:%03d#arguments:%05d\t\r' % (f_count, s_count, p_count, len(tags)))
		sys.stdout.flush()
	print('')
	sentences.append(sentence) # appends last sentence

	print('total sentences',len(sentences))
	print('total propositions',len(predicates))
	print('total arguments',len(tags))
	# print(sentences[0]['ID']) 
	return sentences, predicates, tags 

def print_sentence(sentences, i=None):
	# random.seed(a=13)
	if i==None:
		sentence= random.choice(sentences)		
	else:
		sentence= sentences[i]

	N = len(sentence['ID'])
	# import code; code.interact(local=dict(globals(), **locals()))
	# K = len(sentence['ARG']) / N 
	print('This is an example...')
	# print_header(sentence.keys(), K)		
	print_word(sentence.keys())		
	# for t in range(N+nargs-1):
	for i in range(N):
		words=[]
		for key, word in sentence.items():			
			if word: 
				words.append(word[i])		
				

		print_word(words)		
	

def print_word(values):
	# import code; code.interact(local=dict(globals(), **locals()))
	buff= '%s\t'*len(values)
	print(buff % tuple(values))

def print_header(values, nargs):
	# import code; code.interact(local=dict(globals(), **locals()))
	buff= '%s\t'*(len(values) + (nargs-1))
	if nargs-1>0:
		# import code; code.interact(local=dict(globals(), **locals()))
		buff += 'ARG%i\t' % tuple(range(nargs-1))
	print(buff % tuple(values))


def get_signature(mappings): 
	return {k:[] for k in mappings}


if __name__== '__main__':		
	sentences, predicates, tags= propbankbr_parser()
	print_sentence(sentences,  0)
	print(sentences[0]['PRED'])
	print(sentences[0]['ARG0'])
	print(sentences[0]['ARG1'])
	





