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
import networkx as nx

class FeatureFactory(object):
    # Allowed classes to be created
    @staticmethod
    def klasses():
        return {'ColumnShifter', 'ColumnShifterCTX_P',
        'ColumnPassiveVoice', 'ColumnPredDist', 'ColumnPredMarker', 'ColumnPredMorph',
        'ColumnT'}

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

class ColumnPassiveVoice(object):
    '''
        Passive voice indicator
        1 if POS of target verb GPOS=v-pcp and is preceeded by LEMMA=ser

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
                passive_voice .: dict<PASSIVE_VOICE, OrderedDict<int, int>>
        '''
        # defines output data structure
        self.passive_voice = {'PASSIVE_VOICE': OrderedDict({})}

        # Finds predicate position
        predicate_d = {
            self.dict_db['P'][time]: time
            for time, arg in self.dict_db['ARG'].items() if arg == '(V*)'
        }
        pos_d = {
            self.dict_db['P'][time]: time
            for time, pos in self.dict_db['GPOS'].items() if pos == 'V-PCP'
        }
        lemma_d = {
            self.dict_db['P'][time]: time
            for time, lem in self.dict_db['LEMMA'].items() if lem == 'ser'
        }
        
        for time, proposition in self.dict_db['P'].items():
            predicate_time = predicate_d[proposition]
            lemma_time = lemma_d.get(proposition, None)
            pos_time = pos_d.get(proposition, None)
            if lemma_time and pos_time:
                self.passive_voice['PASSIVE_VOICE'][time] = 1 if lemma_time < predicate_time and pos_time == predicate_time else 0
            else:
                self.passive_voice['PASSIVE_VOICE'][time] = 0 

        return self.passive_voice


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


class ColumnPredMorph(object):
    '''
        Genatate 32 binary array 
        the field MORF is multivalued and is pipe separator ('|')
        Receives 1 if attribute is present 0 otherwise 

        Usage:
            See below (main)
    '''
    def __init__(self, dict_db):
        self.dict_db = dict_db

    def exec(self):
        '''
            Computes 32 item list of zeros and ones
            args:
            returns:
                predmorph .: dict<PRED_MORPH, OrderedDict<int, int>>
        '''
        # defines output data structure
        self.predmorph = {'PRED_MORPH': OrderedDict({})}

        # Finds all single flag
        composite_morph = sorted(list(set(self.dict_db['MORF'].values())))

        morph = [item
                 for sublist in
                 [m.split('|') for m in composite_morph]
                 for item in sublist]

        morph = sorted(list(set(morph)))
        rng = range(len(morph))
        morph2idx = dict(zip(morph, rng))


        for time, morph_comp in self.dict_db['MORF'].items():
            _features = [1 if m in morph_comp.split('|') else 0
                         for m in morph2idx]

            _features = {
                'PredMorph_{:02d}'.format(i + 1):feat_i
                for i, feat_i in enumerate(_features)}

            for key, val in _features.items():
                if key not in self.predmorph['PRED_MORPH']:
                    self.predmorph['PRED_MORPH'][key] = OrderedDict({})
                self.predmorph['PRED_MORPH'][key][time] = val

        return self.predmorph


class ColumnDepFinder(object):
    '''
        Finds columns in Dependency Tree
    '''

    def __init__(self, dict_db):
        self.db = dict_db

    def exec(self):
        '''
            Computes the distance to the target predicate
        '''
        # defines output data structure
        self.kernel = {'HEAD': OrderedDict({})}

        # Finds predicate position
        propositions = sorted(list(set(self.db['P'].values())))
        col = 'FORM'
        prev_prop = -1
        prev_time = -1
        for time, proposition in self.db['P'].items():
            if prev_prop < proposition:
                lb = time
                if prev_prop > 0:
                    process = True 

            if process:
                #Do amazing stuff
                G, root = self._build(lb, prev_time)
                d = self._make_lookupnodes()
                for i in range(lb, prev_time):
                    self._dfs(G, root, i, d)
                    import code; code.interact(local=dict(globals(), **locals()))

            process = False
            prev_prop = proposition
            prev_time = time

        return self.kernel

    def _make_lookupnodes(self):
        _list_keys = ['parent', 'grand_parent', 'child_1', 'child_2', 'child_3']
        return dict.fromkeys(_list_keys)

    def _update_children(self, node, lookup_nodes):
        if lookup_nodes['child_1']:
            if lookup_nodes['child_2']:
                lookup_nodes['child_3'] = node
            else:
                lookup_nodes['child_2'] = node
        else:
            lookup_nodes['child_1'] = node

    def _update_ancestors(self, node, lookup_nodes):
        if lookup_nodes['parent']:
            lookup_nodes['grand_parent'] = lookup_nodes['parent']
        lookup_nodes['parent'] = node

    def _dfs(self, G, u, i, lookup_nodes):
        G[u]['discovered'] = True
        # updates ancestors if target i is undiscovered
        if not G[i]['discovered']:
            self._update_parents(u, lookup_nodes)

        # current node u is target node i
        if i == u:
            for n, v in enumerate(G.neighbors(u)):
                    if not G[v]['discovered']:  # shouldn't be true
                        self._update_children(v, lookup_nodes)
                    if n == 2:
                        return False
        else:
            # keep looking
            for v in G.neighbors(u):
                    if not G[v]['discovered']:
                        search = self._dfs(G, v, i, lookup_nodes)
                        if not search:
                            return False

    def _refresh(self, G):
        for u in G:
            u.discovered = False

    def _build(self, lb, ub):
        G = nx.Graph()
        root = None
        for i in range(lb, ub):
            G.add_node(i, **self._crosssection(i))

        for i in range(lb, ub):
            v = G[i]['DTREE']  # reference to the next node
            u = G[i]['ID']  # reference to the current node within proposition
            G.add_edge(i, (v - u) + i)
            if v == 0:
                root = i
        return G, root

    def _crosssection(self, idx):
        list_keys = list(self.db.keys())
        d = {key: self.db[key][idx] for key in list_keys}
        d['discovered'] = False
        return d





def _process_passivevoice(dictdb):

    pvoice_marker = FeatureFactory().make('ColumnPassiveVoice', dictdb)
    target_dir = '../datasets/csvs/column_passivevoice/'
    passivevoice = pvoice_marker.exec()

    _store(passivevoice, 'passive_voice', target_dir)


def _process_predmorph(dictdb):

    morpher = FeatureFactory().make('ColumnPredMorph', dictdb)
    target_dir = '../datasets/csvs/column_predmorph/'
    predmorph = morpher.exec()

    _store(predmorph['PRED_MORPH'], 'pred_morph', target_dir)


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


def _store(d, target_name, target_dir):
        df = pd.DataFrame.from_dict(d)        
        filename = '{:}{:}.csv'.format(target_dir, target_name)
        df.to_csv(filename, sep=',', encoding='utf-8', index=True)


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
    # columns = ['FORM']
    # delta = 3
    # shifts = [d for d in range(-delta, delta + 1, 1)]
    # _process_shifter_ctx_p(dictdb, columns, shifts)


    # _process_t(dictdb)
    # _process_predicate_marker(dictdb)
    # _process_predmorph(dictdb)
    _process_passivevoice(dictdb)
