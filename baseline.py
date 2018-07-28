from models.propbank_encoder import PropbankEncoder
import config

if __name__ == '__main__':
    encoder_path = '{:}{:}'.format(config.INPUT_DIR, 'deep_glo50')
    propbank_encoder = PropbankEncoder.recover(encoder_path)


    # def iterator(self, ds_type, filter_columns=['P', 'T'], encoding='EMB'):
    columns_ = ['INDEX', 'P', 'FORM', 'GPOS', 'PRED','ARG']
    iter_ = propbank_encoder.iterator('valid', filter_columns=columns_, encoding='CAT')
    first = True

    prev_tag_ = '*'
    prev_prop_ = -1

    baseline_list = []
    gold_list = []
    for index_, prop_, form_, pred_, arg_ in iter_:
        if not (first or prop_ == prev_prop_):
            baseline_list.append(None)
            gold_list.append(None)

            prev_tag_ = '*'

        tag_ = '*'
        # VERB RULE
        if not pred_ == '-':
            # if prev_form_ == 'se':
            #     tag_ = '(C-V*)'
            #     prev_tag_ = '(V*)'
            # else:
            tag_ = '(V*)'
        elif prev_tag_ == '(V*)' and form_ == 'se':
            tag_ = '(C-V*)'

        # NEGATION RULE
        if form_ == 'não':
            tag_ = '(AM-NEG*)'


        if not first:
            gold_list.append((pred_, arg_))
            baseline_list.append((pred_, tag_))

        prev_tag_ = tag_
        prev_prop_ = prop_
        first = False














