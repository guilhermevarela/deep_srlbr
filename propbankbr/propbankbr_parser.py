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


def propbankbr_parser(tokens=[], verbs=[], verbose=True):		
	filename= 	MAPPER['CONST']['filename']
	mappings=   MAPPER['CONST']['mappings']

	sentences1, predicates1, tags1=propbankbr_fileparser(filename, mappings, tokens=[], verbs=[])
	if verbose:
		print_sentence(sentences1,  0)

	filename= 	MAPPER['DEP']['filename']
	mappings=   MAPPER['DEP']['mappings']
	sentences2, predicates2, tags2=propbankbr_fileparser(filename, mappings, sep=' ', tokens=[], verbs=[])	
	if verbose: 
		print_sentence(sentences2,  0)

	return sentences1, predicates1, tags1

def propbankbr_fileparser(filename, mappings, sep='\t',tokens=[], verbs=[]):
	'''
		generates development, validation and test sets according to 1421593_2016_completo
	'''

	# How to represent the dataset?
	# which datastructure best represents the dataset?
	# An array of dictionaries
	# 	each example is a dictionary
	imappings= {v:k for k,v in mappings.items()}

	sentences= [] 	
	f_count=0
	s_count=-1 # sentence counter
	t_count=0  # tags counter
	p_count=0  # predicates count
	sentence=get_signature(mappings)
	predicates={}
	tags={}
	print('#rows\t#sentences\t#predicates\t#arguments\t#tokens\t#verbs')
	for line in open(filename):		
		# import code; code.interact(local=dict(globals(), **locals()))
		# every new sentence except the first is followed by
		if ((f_count == 0) | (line == '\n')):			
			s_count+=1
			if f_count>0: # excludes first
				sentences.append(sentence)
			sentence= get_signature(mappings)

		if len(line)>1:
			# import code; code.interact(local=dict(globals(), **locals()))
			# values= line.split('\t')
			# values= list(map(lambda s: s.replace(' ',''), values))			
			# import code; code.interact(local=dict(globals(), **locals()))
			
			data= line.split(sep)
			data= filter(lambda s: s != '', data)			
			data= map(lambda s: s.replace(' ',''), data)			
			values= list(data)

			# values= list(filter(lambda s: s != '', values))			

			for keys_count, val in enumerate(values): 
				if keys_count in imappings.keys():
					key=imappings[keys_count]	
					
					if (key[:3]=='ARG'):
						val= val.translate(str.maketrans('','','\n()*')) 
						sentence[key].append(val)
					else:
						sentence[key].append(val)

					if (key=='LEMMA'):					
						tokens.append(val)

					if (key=='PRED'):
						if val != '-':
							verbs.append(val)
							predicates[p_count]=s_count 		
							p_count+=1

					if (key[:3] == 'ARG'): # SPECIAL CASE WITH MULTIPLE ARGS																
						#updates arguments	
						if ((val != '') and (val != 'V') and (val != '-')) :
							tags[t_count]={ 
								'pos':(f_count,s_count,p_count),				
								'tag': val
							}
							t_count+=1
							
		f_count+=1		

		sys.stdout.write('%05d\t%05d\t%03d\t%05d\t%05d\t%05d\r' % (f_count, s_count, p_count, len(tags), len(tokens), len(verbs)))
		sys.stdout.flush()
	print('')
	sentences.append(sentence) # appends last sentence

	print('total sentences',len(sentences))
	print('total propositions',len(predicates))
	print('total arguments',len(tags))
	print('tokens:%05d\tunique:%05d\t'% (len(tokens),len(set(tokens))))
	print('verbs:%03d\tunique:%03d\t'% (len(verbs),len(set(verbs))))


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

def get_arguments(tags):
	arguments={}
	for val in tags.values():
		if val['tag'] in arguments:
			arguments[val['tag']]+= 1
		else:
			arguments[val['tag']]=1
	
	print('total tags:', len(tags))
	for key, value in arguments.items():
		print(key, value)
  
	return arguments

def get_arguments_length(sentences):
	arguments_length={}
	for sentence in sentences:
		for key, array in sentence.items():
			if key[:3] == 'ARG' and len(array)>0:
				if key in arguments_length:
					arguments_length[key]+= 1
				else:
					arguments_length[key]=1
	
	for key, value in arguments_length.items():
		print(key, value)
  
	return arguments_length



if __name__== '__main__':		
	tokens=[] 
	sentences, predicates, tags= propbankbr_parser(tokens)
	print(len(tokens))
	print_sentence(sentences,  0)

	stats= get_arguments(tags)
	stats_length= get_arguments_length(sentences)
	





