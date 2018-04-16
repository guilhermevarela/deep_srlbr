'''
    Author Guilherme Varela

    Feature engineering module
    * Converts linguist and end-to-end features into objects
'''


class FeatureFamily(object): 
    # Allowed classes to be created
    @staticmethod
    def klasses():
        return {'ColumnShifter'}
    
    # Creates an instance of class given schema and db
    @staticmethod
    def factory(klass, dict_schema, dict_db):
        if klass == 'ColumnShifter': return ColumnShifter(dict_schema, dict_db)
        raise ValueError('klass must be in {:}'.format(FeatureFamily.klasses))

class ColumnShifter()
    '''
        Shifts columns respecting proposition bounds
    '''
    def __init___(self, dict_schema, dict_db):
        self.dict_schema = dict_schema
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

        '''
        # Check if columns is subset of db columns
        if not(set(columns) <= set(self.dict_db.keys())):
            unknown_columns =  set(columns) - set(self.dict_db.keys())
            raise ValueError('Unknown columns {:}'.format(unknown_columns))

        shift_types = [isinstance(i, int) for i in shifts]
        if not all(shift_types):
            invalid_types =  [shift[i`] for i in shift_types if not shift_types[i]]
            raise ValueError('Unknown columns {:}'.format(invalid_types))

    



    
    def exec(self):
