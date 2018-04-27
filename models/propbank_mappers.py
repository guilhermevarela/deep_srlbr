'''
    Created on Apr 26, 2018
        @author: Varela

    PropbankMappers provides extra functionality for propbank encoder
'''
from collections import OrderedDict
from propbank_encoder import PropbankEncoder
import data_propbankbr as br


class BaseMapper(object):
    _propbank_attributes_ = ()

    def __init__(self,  propbank_encoder):
        for attr in self._propbank_attributes_:
            setattr(self.__class__, attr, getattr(propbank_encoder, attr))


class MapperT2ARG(BaseMapper):
    '''
        Converts a T column in indexed or categorical representation
            into ARG column for CONLL evaluation
    '''
    _propbank_attributes_ = ('idx2lex', 'db')
    def __init__(self, propbank_encoder):
        super().__init__(propbank_encoder)

    def define(self, T, encoding):
        '''
            Converts column T into ARG

            args:
                T .: dict<int, (str, int)> keys in db.keys()
                    type(values) str then use encoding 'CAT'
                    type(values) int then use encoding 'IDX'


                encoding .: str in ('CAT', 'IDX') 
                    'CAT' for categorical representation 'A0', '*', etc
                    'IDX' for indexed representation 1, 3, 4 ~ [0-35]


        '''

        if encoding not in ('CAT', 'IDX'):
            _errmessage = 'encoding must be in {:} got {:}'
            _errmessage = _errmessage .format(('CAT', 'IDX'), encoding)
            raise ValueError(_errmessage)
        else:
            if encoding in ('IDX'):
                T = {idx: self.idx2lex['T'][iidx] for idx, iidx in T.items()}

        self._T = T
        return self

    def map(self):
        '''
            Converts column T into ARG

            args:
                T .: dict<int, str> keys in db_index, values: prediction label

            returns:
                ARG .: dict<int, str> keys in db_index, values in target label
        '''
        db = self.db
        propositions = {idx: db['P'][idx] for idx in self._T}

        ARG = br.propbankbr_t2arg(propositions.values(), self._T.values())
        ordered_dict = sorted(zip(self._T.keys(), ARG), key=lambda x: x[0])
        return OrderedDict(ordered_dict)


class MapperTensor2Column(BaseMapper):
    '''
        Converts a tensor into a propbank column
    '''
    _propbank_attributes_ = ('idx2lex', 'db')
    def __init__(self, propbank_encoder):
        super().__init__(propbank_encoder)

    def define(self, tensor_index, tensor_values, tensor_times, column):
        '''
            Tensors are zero padded numpy arrays i.e :
                tensor.shape ~ [DATABASE_SIZE, MAX_TIME] 
                0 if for t=0,....,MAX_TIME t > times[i] for i=0...DATABASE_SIZE 

            args:
                tensor_index          .: int in db['INDEX']
                tensor_values         .: int in lex2idx[column].values()
                tensor_times         .: int in db['ID'] times or ID column
                column                .: str in db.keys()
        '''
        if column not in self.db:
            buff= '{:} must be in {:}'.format(column, self.db)
            raise KeyError(buff)
        self._tensor_index = tensor_index
        self._tensor_values = tensor_values
        self._tensor_times = tensor_times
        self._tensor_column = column
        return self



    def map(self):
        '''
            Converts a zero padded tensor to a dict

            returns:
                column .: dict<int, str> keys in db_index, values in columns or values in targets
        '''
        tensor_index = self._tensor_index
        tensor_values = self._tensor_values
        times = self._tensor_times
        column = self._tensor_column


        index = [item 
            for i, sublist in enumerate(tensor_index.tolist())
            for j, item in enumerate(sublist) if j < times[i]]

        values = [self.idx2lex[column][item]
            for i, sublist in enumerate(tensor_values.tolist())
            for j, item in enumerate(sublist) if j < times[i]]

        ordered_dict = sorted(zip(index, values), key=lambda x: x[0])
        return OrderedDict(ordered_dict)


class Mapper2SVM(BaseMapper):
    '''
        Maps propbank into a SVM problem file
    '''
    _propbank_attributes_ = ('encodings', 'iterator',
        'columns_config', 'column_dimensions','columns',
        'embeddings', 'embeddings_model', 'embeddings_sz',
        'idx2lex', 'lex2tok', 'tok2idx')
    _implemented_encodings_ = ('EMB', 'HOT')
    _ds_types_ = ['test', 'valid', 'train']
    def __init__(self, propbank_encoder):
        super().__init__(propbank_encoder)

    def define(self, svm_path='../datasets/svms/',
               encoding='EMB', excludecols=['ARG'], target='T'):


        if encoding.upper() not in self.encodings:
            raise ValueError('dump_type must be in {:}'.format(self.encodings))

        if encoding.upper() not in self._implemented_encodings_:
            raise NotImplementedError('You must implement dump_type=={:}'.format(encoding.lower()))

        self.columns = sorted([col for col in self.columns
                        if col not in excludecols and not col == target])

        self.target = target
        self.path = svm_path
        self.encoding = encoding
        return self

    def map(self):

        if self.encoding.upper() in ('EMB'):
            alias = 'word' if self.embeddings_model == 'word2vec' else self.embeddings_model[:3]
            svm_name = 'word' if self.embeddings_model == 'word2vec' else self.embeddings_model[:3]
            svm_name += str(self.embeddings_sz)
        else:
            alias = 'hot'
            svm_name = ''

        filter_columns = self.columns + [self.target]
        for ds_type in self._ds_types_:
            file_name = '{:}_{:}.svm'.format(ds_type, svm_name)
            file_path = '{:}{:}/{:}'.format(self.path, alias, file_name)
            iterator = self.iterator(ds_type, filter_columns=filter_columns, encoding='IDX')
            with open(file_path, mode='w') as f:
                for idx, d in iterator:
                    i = 1
                    line = '{:}'.format(d[self.target])
                    for col in self.columns:
                        sz = self.column_dimensions(col, self.encoding)
                        colconfig = self.columns_config[col]
                        if d[col]:
                            if colconfig['type'] in ('text') and self.encoding.upper() in ('EMB'):
                                emb = self._embeddings_lookup(col, d[col])
                                line += ' '.join(
                                    ['{:}:{:.6f}'.format(i + j, x)
                                     for j, x in enumerate(emb)]
                                ) + ' '
                            elif colconfig['type'] in ('choice') or\
                                (colconfig['type'] in ('text') and self.encoding.upper() in ('HOT')):
                                j = d[col]
                                line += ' {:}:{:}'.format(i + j, 1)
                            else:
                                line += ' {:}:{:}'.format(i, d[col])
                        i += sz + 1

                    line = line + '\n'
                    f.write(line)

    def _embeddings_lookup(self, lookup_column, lookup_key):
        word = self.idx2lex[lookup_column][lookup_key]
        token = self.lex2tok[word]
        idx = self.tok2idx[token]
        return self.embeddings[idx]


if __name__ == '__main__':
    propbank_encoder = PropbankEncoder.recover('../datasets/binaries/deep_glo50.pickle')
    to_svm = Mapper2SVM(propbank_encoder)
    to_svm.define(excludecols=['P','ARG', 'S', 'INDEX']).map()
