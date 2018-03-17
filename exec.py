from config import *
from models.evaluator_conll import EvaluatorConll
from models.propbank import Propbank

if __name__ == '__main__':
	propbank= Propbank.recover(INPUT_DIR + 'db_pt_LEMMA_glove_s50.pickle')
	evaluator= EvaluatorConll()

	S_d= propbank.feature('valid', 'S', True)
	P_d=  propbank.feature('valid', 'P', True)
	PRED_d= propbank.feature('valid', 'PRED', True)
	ARG_d=  propbank.feature('valid', 'ARG', True)

	evaluator.evaluate(S_d, P_d, PRED_d, ARG_d, ARG_d)
	import code; code.interact(local=dict(globals(), **locals()))