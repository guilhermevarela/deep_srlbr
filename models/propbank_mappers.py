'''
    Created on Apr 26, 2018
        @author: Varela

    PropbankMappers provides extra functionality for propbank encoder
'''
from collections import OrderedDict
from .propbank_encoder import PropbankEncoder
import data_propbankbr as br


class MapperT2ARG(PropbankEncoder):
    '''
        Converts a T column in indexed or categorical representation and outputs the arguments
    '''

    def __init__(self, T, encoding):
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

    def map(self):
        '''
            Converts column T into ARG

            args:
                T .: dict<int, str> keys in db_index, values: prediction label

            returns:
                ARG .: dict<int, str> keys in db_index, values in target label
        '''
        propositions = {idx: self.db['P'][idx] for idx in self._T}

        ARG = br.propbankbr_t2arg(propositions.values(), self._T.values())
        ordered_dict = sorted(zip(self._T.keys(), ARG), key=lambda x: x[0])
        return OrderedDict(ordered_dict)


class MapperTensor2Col(PropbankEncoder):
    '''
    '''
    def __init__(self, tensor_index, tensor_values, times, column):
        if column not in self.db:
            buff= '{:} must be in {:}'.format(column, self.db)
            raise KeyError(buff)
        self.__tensor_index = tensor_index
        self.__tensor_values = tensor_values
        self.__tensor_times = times
        self.__tensor_column = column



    def map(self):
        '''
            Converts a zero padded tensor to a dict

            Tensors must have the following shape [DATABASE_SIZE, MAX_TIME] with
                zeros if for t=0,....,MAX_TIME t>times[i] for i=0...DATABASE_SIZE 

        args:
            tensor_index  .: with db index

            tensor_values .: with db int values representations

            times  .: list<int> [DATABASE_SIZE] holding the times for each proposition

            column .: str           db column name

        returns:
            column .: dict<int, str> keys in db_index, values in columns or values in targets
        '''

        index=[item  for i, sublist in enumerate(tensor_index.tolist())
            for j, item in enumerate(sublist) if j < times[i]]

        values = [self.idx2lex[column][item]
            for i, sublist in enumerate(tensor_values.tolist())
            for j, item in enumerate(sublist) if j < times[i]]

        return dict(zip(index, values))
