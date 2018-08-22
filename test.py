from models.evaluator_conll import EvaluatorConll

if __name__ == '__main__':
	gold_path = 'lr5.00e-03_hs32_ctx-p1/best-valid-gold.props'
	predicted_path = 'lr5.00e-03_hs32_ctx-p1/best-valid-eval.props' 

	f1 = EvaluatorConll.evaluate_frompropositions(gold_path, predicted_path,
						     'random-test', verbose=True, keep_list=['A0', 'A1', 'AM-NEG'])
