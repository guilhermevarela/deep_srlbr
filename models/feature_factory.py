'''
    Author Guilherme Varela

    Feature engineering module
    * Converts linguist and end-to-end features into objects
'''
from collections import OrderedDict

import pandas as pd


class FeatureFactory(object):
    # Allowed classes to be created
    @staticmethod
    def klasses():
        return {'ColumnShifter'}

    # Creates an instance of class given schema and db
    @staticmethod
    def make(klass, dict_db):
        if klass == 'ColumnShifter': return ColumnShifter(dict_db)
        raise ValueError('klass must be in {:}'.format(FeatureFamily.klasses))


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


if __name__ == '__main__':
    '''
        Usage of FeatureFactory
    '''

    df = pd.read_csv('../datasets/csvs/gs.csv', index_col=0, encoding='utf-8')
    dictdb = df.to_dict()
    shifter = FeatureFactory().make('ColumnShifter', dictdb)

    delta = 3
    # columns = ['FORM', 'LEMMA', 'GPOS']
    columns = ['FUNC']
    shifts = [d for d in range(-delta, delta+1, 1) if d != 0]
    shifted = shifter.define(columns, shifts).exec()
    target_dir = '../datasets/csvs/gs_column_shifts/'
    for col in columns:
        d = {new_col: shifted[new_col]
             for new_col in shifted if col in new_col}
        df = pd.DataFrame.from_dict(d)
        filename = '{:}{:}.csv'.format(target_dir, col.lower())
        df.to_csv(filename, sep=',', encoding='utf-8')
