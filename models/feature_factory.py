'''
    Author Guilherme Varela

    Feature engineering module
    * Converts linguist and end-to-end features into objects
'''
import sys
sys.path.append('../datasets')
from collections import OrderedDict

import data_propbankbr as br
import pandas as pd
import yaml

class FeatureFactory(object):
    # Allowed classes to be created
    @staticmethod
    def klasses():
        return {'ColumnShifter', 'ColumnShifterCTX_P', 'ColumnPredDist', 'ColumnPredMarker', 'ColumnT'}

    # Creates an instance of class given schema and db
    @staticmethod
    def make(klass, dict_db):
        if klass in FeatureFactory.klasses():
            return eval(klass)(dict_db)
        else:
            raise ValueError('klass must be in {:}'.format(FeatureFactory.klasses()))


class ColumnShifter(object):
    '''
        Shifts columns respecting proposition bounds

        Usage:
            See below (main)
    '''
    def __init__(self, dict_db):
        self.dict_db = dict_db

    # Columns over which we want to perform the shifting        
    def define(self, columns, shifts):
        '''
            Defines with columns will be effectively shifted and by what amount

            args:
                columns .: list<str> column names that will be shifted

                shifts .: list<int> of size m holding integers, negative numbers are delays
                    positive numbers are leads

                new_columns .: list<str> of size n holding new column names if none 
                                one name will be generated

            returns:
                column_shifter .: object<ColumnShifter> an instance of column shifter

        '''
        # Check if columns is subset of db columns
        if not(set(columns) <= set(self.dict_db.keys())):
            unknown_columns = set(columns) - set(self.dict_db.keys())
            raise ValueError('Unknown columns {:}'.format(unknown_columns))
        else:
            self.columns = columns

        shift_types = [isinstance(i, int) for i in shifts]
        if not all(shift_types):

            invalid_types = [shift[i] for i in shift_types
                             if not shift_types[i]]

            raise ValueError('Int type violation: {:}'.format(invalid_types))
        else:
            self.shifts = sorted(shifts)

        self.mapper = OrderedDict(
                {(i, col): '{:}{:+d}'.format(col, i)
                 for col in columns for i in sorted(shifts)})

        return self

    def exec(self):
        '''
            Computes column shifting
            args:
            returns:
                shifted .: dict<new_columns, dict<int, column<type>>>
        '''
        if not ( self.columns or self.shifts or self.mapper):
            raise Exception('Columns to be shifted are undefined run column_shifter.define')

        # defines output data structure
        self.dict_shifted = {col: OrderedDict({}) for _, col in self.mapper.items()}

        # defines output data structure

        for time, proposition in self.dict_db['P'].items():
            for col in self.columns:                
                for s in self.shifts:
                    new_col = self.mapper[(s, col)]
                    if (time + s in self.dict_db['P']) and\
                         (self.dict_db['P'][time + s] == proposition):
                        self.dict_shifted[new_col][time] = self.dict_db[col][time + s]
                    else:
                        self.dict_shifted[new_col][time] = None

        return self.dict_shifted

class ColumnShifterCTX_P(object):
    '''
        Grabs columns around predicate and shifts it

        Usage:
            See below (main)
    '''

    def __init__(self, dict_db):
        self.dict_db = dict_db

    # Columns over which we want to perform the shifting    
    def define(self, columns, shifts):
        '''
            Defines with columns will be effectively shifted and by what amount

            args:
                columns .: list<str> column names that will be shifted

                shifts .: list<int> of size m holding integers, negative numbers are delays
                    positive numbers are leads

                new_columns .: list<str> of size n holding new column names if none 
                                one name will be generated

            returns:
                column_shifter .: object<ColumnShifter> an instance of column shifter

        '''
        # Check if columns is subset of db columns
        if not(set(columns) <= set(self.dict_db.keys())):
            unknown_columns = set(columns) - set(self.dict_db.keys())
            raise ValueError('Unknown columns {:}'.format(unknown_columns))
        else:
            self.columns = columns

        shift_types = [isinstance(i, int) for i in shifts]
        if not all(shift_types):
            invalid_types = [shift[i] for i in shift_types if not shift_types[i]]

            raise ValueError('Int type violation: {:}'.format(invalid_types))
        else:
            self.shifts = sorted(shifts)

        if not '(V*)' in set(self.dict_db['ARG'].values()):
            raise ValueError('(V*) not in ARG')

        self.mapper = OrderedDict(
                {(i, col): '{:}_CTX_P{:+d}'.format(col, i)
                 for col in columns for i in sorted(shifts)})

        return self

    def exec(self):
        '''
            Computes column shifting
            args:
            returns:
                shifted .: dict<new_columns, dict<int, column<type>>>
        '''
        if not ( self.columns or self.shifts or self.mapper):
            raise Exception('Columns to be shifted are undefined run column_shifter.define')

        # defines output data structure
        self.dict_shifted = {col: OrderedDict({}) for _, col in self.mapper.items()}

        # defines output data structure
        times = []
        predicate_d = {
            self.dict_db['P'][time]: time
            for time, arg in self.dict_db['ARG'].items() if arg == '(V*)'
        }
        for time, proposition in self.dict_db['P'].items():
            predicate_time =  predicate_d[proposition]
            for col in self.columns:
                for s in self.shifts:
                    new_col = self.mapper[(s, col)]                    
                    if (predicate_time + s in self.dict_db['P']) and\
                         (self.dict_db['P'][predicate_time + s] == proposition):
                        self.dict_shifted[new_col][time] = self.dict_db[col][predicate_time + s]
                    else:
                        self.dict_shifted[new_col][time] = None

        return self.dict_shifted


class ColumnPredDist(object):
    '''
        Computes the distance to the predicate

        Usage:
            See below (main)
    '''
    def __init__(self, dict_db):
        self.dict_db = dict_db

    # Columns over which we want to perform the shifting
    def define(self):
        '''
            Defines with columns will be effectively shifted and by what amount

            args:
                columns .: list<str> column names that will be shifted

                shifts .: list<int> of size m holding integers, negative numbers are delays
                    positive numbers are leads

                new_columns .: list<str> of size n holding new column names if none 
                                one name will be generated

            returns:
                column_shifter .: object<ColumnShifter> an instance of column shifter

        '''
        if not '(V*)' in set(self.dict_db['ARG'].values()):
            raise ValueError('(V*) not in ARG')

        return self

    def exec(self):
        '''
            Computes the distance to the target predicate
            args:
            returns:
                preddist .: dict<PRED_DIST, OrderedDict<int, int>>
        '''
        # defines output data structure
        self.preddist = {'PRED_DIST': OrderedDict({})}

        # Finds predicate position
        predicate_d = {
            self.dict_db['P'][time]: time
            for time, arg in self.dict_db['ARG'].items() if arg == '(V*)'
        }
        for time, proposition in self.dict_db['P'].items():
            predicate_time = predicate_d[proposition]

            self.preddist['PRED_DIST'][time] = predicate_time - time

        return self.preddist


class ColumnT(object):
    '''
        Computes a "simplified" version of ARG
        Removes chunking from ARG

        Usage:
            See below (main)
    '''

    def __init__(self, dict_db):
        self.dict_db = dict_db

    def exec(self):
        '''
            Computes the distance to the target predicate
            args:
            returns:
                preddist .: dict<PRED_DIST, OrderedDict<int, int>>
        '''
        # defines output data structure
        T = br.propbankbr_arg2t(self.dict_db['P'].values(), self.dict_db['ARG'].values())

        self.t = {'T': OrderedDict(
            dict(zip(self.dict_db['ARG'].keys(), T))
        )}

        return self.t


class ColumnPredMarker(object):
    '''
        Marks if we are in the predicate context
        1 if time > predicate_time
        0 otherwise

        Usage:
            See below (main)
    '''

    def __init__(self, dict_db):
        self.dict_db = dict_db

    def exec(self):
        '''
            Computes the distance to the target predicate
            args:
            returns:
                preddist .: dict<PRED_DIST, OrderedDict<int, int>>
        '''
        # defines output data structure
        self.predmarker = {'PRED_MARKER': OrderedDict({})}

        # Finds predicate position
        predicate_d = {
            self.dict_db['P'][time]: time
            for time, arg in self.dict_db['ARG'].items() if arg == '(V*)'
        }
        for time, proposition in self.dict_db['P'].items():
            predicate_time = predicate_d[proposition]

            self.predmarker['PRED_MARKER'][time] = 0 if predicate_time - time > 0 else 1

        return self.predmarker


def _process_shifter(dictdb, columns, shifts):

    shifter = FeatureFactory().make('ColumnShifter', dictdb)
    target_dir = '../datasets/csvs/column_shifter/'
    shifted = shifter.define(columns, shifts).exec()

    _store_columns(shifted, columns, target_dir)


def _process_shifter_ctx_p(db, columns, shifts):

    shifter = FeatureFactory().make('ColumnShifterCTX_P', dictdb)
    target_dir = '../datasets/csvs/column_shifts_ctx_p/'
    shifted = shifter.define(columns, shifts).exec()

    _store_columns(shifted, columns, target_dir)


def _process_predicate_dist(dictdb):

    pred_dist = FeatureFactory().make('ColumnPredDist', dictdb)
    d = pred_dist.define().exec()

    target_dir = '../datasets/csvs/column_preddist/'
    filename = '{:}{:}.csv'.format(target_dir, 'predicate_distance')
    pd.DataFrame.from_dict(d).to_csv(filename, sep=',', encoding='utf-8')


def _process_t(dictdb):

    column_t = FeatureFactory().make('ColumnT', dictdb)
    d = column_t.exec()

    target_dir = '../datasets/csvs/column_t/'
    filename = '{:}{:}.csv'.format(target_dir, 't')
    pd.DataFrame.from_dict(d).to_csv(filename, sep=',', encoding='utf-8')

def _process_predicate_marker(dictdb):

    column_t = FeatureFactory().make('ColumnPredMarker', dictdb)
    d = column_t.exec()

    target_dir = '../datasets/csvs/column_predmarker/'
    filename = '{:}{:}.csv'.format(target_dir, 'predicate_marker')
    pd.DataFrame.from_dict(d).to_csv(filename, sep=',', encoding='utf-8')


def _store_columns(columns_dict, columns, target_dir):
    for col in columns:
        d = {new_col: columns_dict[new_col]
             for new_col in columns_dict if col in new_col}

        df = pd.DataFrame.from_dict(d)
        filename = '{:}{:}.csv'.format(target_dir, col.lower())
        df.to_csv(filename, sep=',', encoding='utf-8')


if __name__ == '__main__':
    '''
        Usage of FeatureFactory
    '''
    df = pd.read_csv('../datasets/csvs/gs.csv', index_col=0, encoding='utf-8')
    dictdb = df.to_dict()

    # Making column moving windpw around column
    # columns = ('FORM', 'LEMMA', 'FUNC', 'GPOS')
    # delta = 3
    # shifts = [d for d in range(-delta, delta + 1, 1) if d != 0]
    # _process_shifter(dictdb, columns, shifts)


    # Making window around predicate
    # columns = ('FUNC', 'GPOS', 'LEMMA', 'FORM')
    # delta = 1
    # shifts = [d for d in range(-delta, delta + 1, 1)]
    # _process_shifter_ctx_p(dictdb, columns, shifts)


    # Computing the distance to target predicate
    # import code; code.interact(local=dict(globals(), **locals()))

    # pred_dist = FeatureFactory().make('ColumnPredDist', dictdb)
    # d = pred_dist.define().exec()
    # target_dir = '../datasets/csvs/column_preddist/'
    # filename = '{:}{:}.csv'.format(target_dir, 'predicate_distance')
    # pd.DataFrame.from_dict(d).to_csv(filename, sep=',', encoding='utf-8')
    _process_t(dictdb)
    _process_predicate_marker(dictdb)
