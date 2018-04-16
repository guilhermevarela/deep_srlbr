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
    def define(self, columns, shifts, new_columns=None):
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

        if (new_columns) and (len(new_columns) != len(columns) * len(shifts)):
            raise ValueError('New columns must be TypeNone or len(columns)*len(shifts)')
        else:
            if new_columns:
                self.new_columns = columns
            else:
                self.new_columns = ['{:}_{:+d}'.format(col, i)
                                    for col in columns for i in sorted(shifts)]
        return self

    def exec(self):
        '''
            Computes column shifting
            args:
            returns:
                shifted .: dict<new_columns, dict<int, column<type>>>
        '''
        if not ( self.columns or self.shifts or self.new_columns):
            raise Exception('Columns to be shifted are undefined run column_shifter.define')

        # defines output data structure
        self.dict_shifted = {col: OrderedDict({}) for col in self.new_columns}

        # defines output data structure
        # current_proposition = 1
        for time, proposition in self.dict_db['P'].items():            
            for i, col in enumerate(self.columns):
                new_col = self.new_columns[i]
                for s in self.shifts:
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
    shifted = shifter.define(['FORM'], [-1, 1]).exec()
    import code; code.interact(local=dict(globals(), **locals()))
