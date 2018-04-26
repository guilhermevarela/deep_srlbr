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


class MapperTensor2Column(PropbankEncoder):
    '''
    '''
    def __init__(self, tensor_index, tensor_values, tensor_times, column):
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
        self.__tensor_index = tensor_index
        self.__tensor_values = tensor_values
        self.__tensor_times = times
        self.__tensor_column = column



    def map(self):
        '''
            Converts a zero padded tensor to a dict

            returns:
                column .: dict<int, str> keys in db_index, values in columns or values in targets
        '''
        tensor_index = self.__tensor_index
        tensor_values = self.__tensor_values
        times = self.__tensor_times
        column = self.__tensor_column

        index = [item 
            for i, sublist in enumerate(tensor_index.tolist())
            for j, item in enumerate(sublist) if j < times[i]]

        values = [self.idx2lex[column][item]
            for i, sublist in enumerate(tensor_values.tolist())
            for j, item in enumerate(sublist) if j < times[i]]

        ordered_dict = sorted(zip(index, values), key=lambda x: x[0])
        return OrderedDict(ordered_dict)
