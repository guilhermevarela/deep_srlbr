'''
    Created on Apr 26, 2018
        @author: Varela

    PropbankMappers provides extra functionality for propbank encoder
'''
import sys
sys.path.append('..')
from collections import OrderedDict
from models.propbank_encoder import PropbankEncoder
import datasets.data_propbankbr as br


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


class MapperIDX2CAT(BaseMapper):
    '''
        Maps index 2 categorical
    '''
    _propbank_attributes_ = ('columns', 'idx2lex', 'lex2tok')
    _ds_types_ = ['test', 'valid', 'train']

    def __init__(self, propbank_encoder):
        super().__init__(propbank_encoder)

    def define(self, d, column):
        if column not in self.columns:
            raise ValueError('column {:} must be in {:}')
        else:
            self.column = column
        self.d = d
        return self

    def map(self):
        return [self.idx2lex[self.column][int(val)]
                for _, val in self.d.items()]
