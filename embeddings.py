from gensim.models import KeyedVectors
from gensim import corpora, models, similarities

if __name__ == '__main__':
	# import code; code.interact(local=dict(globals(), **locals()))			
	w2v = KeyedVectors.load_word2vec_format('datasets/glove_s50.txt', unicode_errors="ignore")
	w2v.most_similar(positive=['mulher', 'rei'], negative=['homem'])
	print(w2v['rainha'])
	corpus= [['o rato roeu a roupa do rei'] ,['o rato roeu a porta']]
	



# In gensim a corpus is simply an object which, when iterated over, 
# returns its documents represented as sparse vectors. In this case we’re using a list of list of tuples
	# corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
	# 	[(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
	# 	[(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
	# 	[(0, 1.0), (4, 2.0), (7, 1.0)],
	# 	[(3, 1.0), (5, 1.0), (6, 1.0)],
	# 	[(9, 1.0)],
	# 	[(9, 1.0), (10, 1.0)],
	# 	[(9, 1.0), (10, 1.0), (11, 1.0)],
	# 	[(8, 1.0), (10, 1.0), (11, 1.0)]]
