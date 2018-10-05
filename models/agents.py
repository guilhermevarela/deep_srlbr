'''
Created on Oct 4, 2018
    @author: Varela

    Agents act as layer of abstraction between client
    and tensorflow computation grapth
'''
import os
import tensorflow as tf

import config
from models.conll_evaluator import ConllEvaluator
from models.propbank_encoder import PropbankEncoder
from models.labelers import Labeler
from models.streamers import TfStreamer

from utils.snapshots import snapshot_hparam_string, snapshot_persist, \
    snapshot_recover

from utils.info import get_binary

FEATURE_LABELS = ['ID', 'FORM', 'MARKER', 'GPOS',
                  'FORM_CTX_P-1', 'FORM_CTX_P+0', 'FORM_CTX_P+1']

TARGET_LABEL = ['IOB']

HIDDEN_LAYERS = [16, 16]



class AgentMeta(type):
    '''This is a metaclass -- enforces method definition
    on function body

    Every agent must implent the following methods
    * evaluate -- evaluates a previously defined model
    * fit -- trains a model
    * load -- loads a model from disk
    * predict -- predicts the outputs from numpy matrices
    * predict_fromfile -- predicts the outputs based on stored data
    * _build_cgraph -- builds the computation graph after
                      inialization

    References:
        https://docs.python.org/3/reference/datamodel.html#metaclasses
        https://realpython.com/python-metaclasses/
    '''
    def __new__(meta, name, base, body):
        agent_methods = ('evaluate', 'fit', 'load', 'predict',
                         'predict_fromfile', '_build_graph')

        for am in agent_methods:
            if am not in body:
                msg = 'Agent must implement {:}'.format(am)
                raise TypeError(msg)

        return super().__new__(meta, name, base, body)



class SrlAgent(metaclass=AgentMeta):
    '''[summary]

    [description]

    Extends:
        metaclass=AgentMeta
    '''
    def __init__(self, input_labels=FEATURE_LABELS, target_labels=TARGET_LABEL,
                 hidden_layers=HIDDEN_LAYERS, embeddings_model='wan50',
                 embeddings_trainable=False, epochs=100, lr=5 * 1e-3,
                 batch_size=250, ctx_p=1, version='1.0',
                 ru='BasicLSTM', chunks=False, r_depth=-1, **kwargs):

        # ckpt_dir should be set by SrlAgent#load
        ckpt_dir = kwargs.get('ckpt_dir', None)
        kfold = kwargs.get('kfold', False)
        if ckpt_dir is None:
            target_dir = snapshot_hparam_string(
                embeddings_model=embeddings_model,
                target_labels=target_labels,
                is_batch=not kfold, ctx_p=ctx_p,
                learning_rate=lr, version=version,
                hidden_layers=hidden_layers)

            target_dir = 'outputs{:}'.format(target_dir)
            target_dir = snapshot_persist(
                target_dir,
                input_labels=input_labels, lr=lr,
                hidden_layers=hidden_layers, ctx_p=ctx_p,
                target_labels=target_labels, kfold=25,
                embeddings_trainable=False,
                embeddings_model=embeddings_model, ru=ru,
                epochs=epochs, chunks=chunks, r_depth=r_depth,
                version=version)
        else:
            self.target_dir = ckpt_dir


        self._dyn_attr(locals())
        self._build_graph()

    @classmethod
    def load(cls, ckpt_dir):
        # self.target_dir = ckpt_dir
        # self.session_path = '{:}model.ckpt'.format(ckpt_dir)
        with open(ckpt_dir + 'params.json', mode='r') as f:
            attr_dict = json.load(f)

        # prevent from creating a new directory
        attr_dict['ckpt_dir'] = ckpt_dir
        agent = cls(**attr_dict)
        agent.target_dir = ckpt_dir
        agent.session_path = '{:}model.ckpt'.format(ckpt_dir)

        return agent

    def evaluate(self, I, Y, L, filename):
        f1 = None
        try:
            f1 = self.evaluator.evaluate_npyarray(
                filename, I, Y, L, self.target_labels, {}, script_version='04'
            )
        except AttributeError:
            err = '''evaluator not defined -- either re start a new instance
                     or load from {:}'''.format(self.target_dir)
            raise AttributeError(err)
        finally:
            return f1

    def fit(self):
        X_train, T_train, L_train, I_train = TfStreamer.get_train(
            self.input_labels, self.target_labels, version=self.version,
            embeddings_model=self.embeddings_model
        )

        X_valid, T_valid, L_valid, I_valid = TfStreamer.get_valid(
            self.input_labels, self.target_labels, version=self.version,
            embeddings_model=self.embeddings_model
        )

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        with tf.Session() as session:
            self.session = session
            session.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            saver = tf.train.Saver()
            session_path = '{:}model.ckpt'.format(self.target_dir)
            # tries to restore saved model
            if os.path.isfile(session_path):
                saver.restore(session, self.target_dir)
                conll_path = '{:}best-valid.conll'.format(self.target_dir)
                self.evaluator.evaluate_fromconllfile(conll_path)
                best_validation_rate = evaluator.f1
            else:
                best_validation_rate = -1

            # Training control variables
            step = 1
            total_loss = 0.0
            total_error = 0.0
            eps = 100
            try:
                while not (coord.should_stop() or eps < 1e-3):
                    X_batch, T_batch, L_batch, I_batch = session.run(self.streamer.stream)
                    # import code; code.interact(local=dict(globals(), **locals()))

                    loss, _, Yish, error = session.run(
                        [self.deep_srl.cost, self.deep_srl.label, self.deep_srl.predict, self.deep_srl.error],
                        feed_dict={self.X: X_batch, self.T: T_batch, self.L: L_batch}
                    )

                    total_loss += loss
                    total_error += error
                    if (step) % 25 == 0:

                        # Y_train = session.run(
                        #     self.deep_srl.predict,
                        #     feed_dict={self.X: X_train, self.T: T_train, self.L: L_train}
                        # )
                        _, f1_train = self.predict(I_train, X_train, L_train, evalfilename='train')
                        # f1_train = self.evaluate(Y_train, I_train, 'train')

                        # Y_valid = session.run(
                        #     self.deep_srl.predict,
                        #     feed_dict={self.X: X_valid, self.T: T_valid, self.L: L_valid}
                        # )
                        # f1_valid = self.evaluate(Y_valid, I_valid, L_valid, 'valid')
                        
                        Y_valid, f1_valid = self.predict(I_valid, X_valid, L_valid, evalfilename='valid')
                        if f1_valid is not None and f1_train is not None:
                            print('Iter={:5d}'.format(step),
                                  '\tavg. cost {:.6f}'.format(total_loss / 25),
                                  '\tavg. error {:.6f}'.format(total_error / 25),
                                  '\tf1-train {:.6f}'.format(f1_train),
                                  '\tf1-valid {:.6f}'.format(f1_valid))
                        else:
                            print('Iter={:5d}'.format(step),
                                  '\tavg. cost {:.6f}'.format(total_loss / 25),
                                  '\tavg. error {:.6f}'.format(total_error / 25))
                        eps = float(total_error) / 25
                        total_loss = 0.0
                        total_error = 0.0

                        if f1_valid and best_validation_rate < f1_valid:
                            best_validation_rate = f1_valid
                            f1_valid = self.evaluate(I_valid, Y_valid, L_valid, 'best-valid')
                            saver.save(session, session_path)
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

            finally:
                # When done, ask threads to stop
                coord.request_stop()
                coord.join(threads)

    def predict(self, I, X, L, evalfilename=None):
        Y = self.session.run(
            self.deep_srl.predict,
            feed_dict={self.X: X, self.L: L})

        if evalfilename is None:
            return Y
        else:
            f1 = self.evaluate(I, Y, L, evalfilename)
            return Y, f1


    def predict_fromfile(self):
        pass


    def _build_graph(self):
        propbank_path = get_binary(
            'deep', self.embeddings_model, version=self.version)
        propbank_encoder = PropbankEncoder.recover(propbank_path)

        dataset_path = get_binary(
            'train', self.embeddings_model, version=self.version)
        datasets_list = [dataset_path]

        self.evaluator = ConllEvaluator(propbank_encoder,
                                        target_dir=self.target_dir)

        cnf_dict = config.get_config(self.embeddings_model)
        X_shape = get_xshape(self.input_labels, cnf_dict)
        T_shape = get_tshape(self.target_labels, cnf_dict)
        print('trainable embeddings?', self.embeddings_trainable)
        print('X_shape', X_shape)
        print('T_shape', T_shape)

        self.X = tf.placeholder(tf.float32, shape=X_shape, name='X')
        self.T = tf.placeholder(tf.float32, shape=T_shape, name='T')
        self.L = tf.placeholder(tf.int32, shape=(None,), name='L')



        # The streamer instanciation builds a feeder_op that will
        # supply the computation graph with batches of examples
        self.streamer = TfStreamer(datasets_list, self.batch_size, self.epochs,
                                   self.input_labels, self.target_labels,
                                   shuffle=True)

        # The Labeler instanciation will build the archtecture
        targets_size = [cnf_dict[lbl]['size'] for lbl in self.target_labels]
        kwargs = {'learning_rate': self.lr, 'hidden_size': self.hidden_layers,
                  'targets_size': targets_size, 'ru': self.ru}
        self.deep_srl = Labeler(self.X, self.T, self.L, **kwargs)

    def _dyn_attr(self, attr_dict):

        def filt(x):
            return (x in ('self','ckpt_dir') or x[0] == '_')

        attr_dict = {n: v for n, v in attr_dict.items() if not filt(n)}

        for k, v in attr_dict.items():
            if not hasattr(self, k):
                setattr(self, k, v)

def get_xshape(input_labels, cnf_dict):
    # axis 0 --> examples
    # axis 1 --> max time
    # axis 2 --> feature size
    xshape = [None, None]
    feature_sz = sum([cnf_dict[lbl]['size'] for lbl in input_labels])
    xshape.append(feature_sz)
    return xshape


def get_tshape(output_labels, cnf_dict):
    # axis 0 --> examples
    # axis 1 --> max time
    base_shape = [None, None]
    k = len(output_labels)
    if k == 1:
        # axis 2 --> target size
        m = cnf_dict[output_labels[0]]['size']
        tshape = base_shape + [m]
    elif k == 2:
        # axis 2 --> max target size
        # axis 3 --> number of targets
        m = max([cnf_dict[lbl]['size'] for lbl in output_labels])
        tshape = base_shape + [m, k]
    else:
        err = 'len(target_labels) <= 2 got {:}'.format(k)
        raise ValueError(err)
    return tshape
