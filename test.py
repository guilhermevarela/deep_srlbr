from models.evaluator_conll import EvaluatorConll

if __name__ == '__main__':
    best_path = 'datasets/best/wan300_iob/lr5.00e-03_hs16x16x16_ctx-p1'
    gold_path = '{:}/best-valid-gold.props'.format(best_path)
    predicted_path = '{:}/best-valid-eval.props'.format(best_path)
    script_version = '04'

    f1 = EvaluatorConll.evaluate_frompropositions(
        gold_path, predicted_path, 'random-test',
        verbose=True, script_version=script_version,
        keep_list=['A0', 'A1', 'AM-NEG'])
