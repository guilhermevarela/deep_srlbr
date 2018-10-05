'''
Created on Oct 4, 2018
    @author: Varela

    Agents act as layer of abstraction between client
    and tensorflow computation grapth
'''
import json
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

TARGET_LABELS = ['IOB']

HIDDEN_LAYERS = [16, 16]



class AgentMeta(type):
    '''This is a metaclass -- enforces method definition
    on function body

    Every agent must implent the following methods
    * evaluate -- evaluate the task using matrices
    * evaluate_dataset -- evaluates the task using dasets from disk
    * fit -- trains a model
    * load -- loads a model from disk
    * predict -- predicts the outputs from numpy matrices


    References:
        https://docs.python.org/3/reference/datamodel.html#metaclasses
        https://realpython.com/python-metaclasses/
    '''
    def __new__(meta, name, base, body):
        agent_methods = ('evaluate', 'fit', 'load', 'predict',
                         'evaluate_dataset')

        for am in agent_methods:
            if am not in body:
                msg = 'Agent must implement {:}'.format(am)
                raise TypeError(msg)

        return super().__new__(meta, name, base, body)



class SRLAgent(metaclass=AgentMeta):
    '''Semantic Role Labeling ST 2004 - 2005

    Defines a tensorflow DataFlow graph using Recurrent Neural Networks,
    the data is fed to the graph using protobuf binaries and uses
    official perl scripts to evaluate the training progress. It needs
    a directory to store the best validation parameter and the evaluations


    Example:
        # Uses SGD to train a labeler evaluating on a
        # separate set using using evaluation scripts
        > srl = SRLAgent()
        > srl.fit()

        # Loads a pre-trained model and evaluates it
        > ckpt_dir = ... # Dir containing model.ckpt.xpto
        > srl = SRLAgent.load(ckpt_dir) # loads a previously trained model
        > srl.evaluate_dataset('valid') #evaluates the dataset
        > srl.fit() # Retrains the dataset from previous session


    Extends:
        metaclass=AgentMeta

    References:
        Jie Zhou and Wei Xu. 2015.
        "End-to-end learning of semantic role labeling using recurrent neural
        networks". In Proc. of the Annual Meeting of the Association
        for Computational Linguistics (ACL)

        http://www.aclweb.org/anthology/P15-1109

        Xavier Carreras and Lluís Màrquez. 2004.
        "Introduction to the CoNLL-2004 Shared Task: Semantic Role Labeling".
        In proccedings of CoNLL 2004.

        https://www.cs.upc.edu/~srlconll/st04/st04.html

    TODO:
        Include raw text inputs to evaluate the model
    '''
    def __init__(self, input_labels=FEATURE_LABELS, target_labels=TARGET_LABELS,
                 hidden_layers=HIDDEN_LAYERS, embeddings_model='wan50',
                 embeddings_trainable=False, epochs=100, lr=5 * 1e-3,
                 batch_size=250, ctx_p=1, version='1.0',
                 ru='BasicLSTM', chunks=False, r_depth=-1, **kwargs):
        '''Defines Dataflow graph

        Builds a Rnn tensorflow graph

        Arguments:
            **kwargs {[type]} -- [description]

        Keyword Arguments:
            input_labels {list} -- Features to be considered
                                    (default: {FEATURE_LABELS})

            target_labels {list} -- Targets more than one label is possible
                                        (default: {TARGET_LABELS})

            hidden_layers {list} -- Integers holding the hidden layers sizes
                                        (default: {HIDDEN_LAYERS})

            embeddings_model {str} -- Abbrev. of embedding_model name and
                                     it's size: GloVe size 50 --> glo50
                                     (default: {'wan50'})

            embeddings_trainable {bool} -- TODO: allow trainable embeddings
                                    (default: {False})

            epochs {int} -- Iterations to make on the training set
                                (default: {100})

            lr {float} -- Learning Rate (default: {5 * 1e-3})

            batch_size {int} -- Number of examples to be trained
                                (default: {250})

            ctx_p {int} -- size of the windoew around the predicate
                                (default: {1})

            version {str} -- Propbank version (default: {'1.0'})

            ru {int} -- Recurrent unit to use (default: {'BasicLSTM'})

            chunks {bool} --  (default: {False})

            r_depth {number} -- [description] (default: {-1})
        '''

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
            self.target_dir = target_dir
            self._restore_session = False
        else:
            self.target_dir = ckpt_dir
            self._restore_session = True

        self.input_labels = input_labels
        self.target_labels = target_labels
        self.version = version
        self.embeddings_model = embeddings_model

        propbank_path = get_binary(
            'deep', embeddings_model, version=version)
        propbank_encoder = PropbankEncoder.recover(propbank_path)

        dataset_path = get_binary(
            'train', embeddings_model, version=version)
        datasets_list = [dataset_path]

        self.evaluator = ConllEvaluator(propbank_encoder, target_dir=self.target_dir)

        cnf_dict = config.get_config(embeddings_model)
        X_shape = get_xshape(input_labels, cnf_dict)
        T_shape = get_tshape(target_labels, cnf_dict)
        print('trainable embeddings?', embeddings_trainable)
        print('X_shape', X_shape)
        print('T_shape', T_shape)

        self.X = tf.placeholder(tf.float32, shape=X_shape, name='X')
        self.T = tf.placeholder(tf.float32, shape=T_shape, name='T')
        self.L = tf.placeholder(tf.int32, shape=(None,), name='L')



        # The streamer instanciation builds a feeder_op that will
        # supply the computation graph with batches of examples
        self.streamer = TfStreamer(datasets_list, batch_size, epochs,
                                   input_labels, target_labels,
                                   shuffle=True)

        # The Labeler instanciation will build the archtecture
        targets_size = [cnf_dict[lbl]['size'] for lbl in target_labels]
        kwargs = {'learning_rate': lr, 'hidden_size': hidden_layers,
                  'targets_size': targets_size, 'ru': ru}
        self.rnn_srl = Labeler(self.X, self.T, self.L, **kwargs)

    @classmethod
    def load(cls, ckpt_dir):
        '''Loads from ckpt_dir a previous experiment

        Loads a pre-trained session either to evaluated or
        to be retrained

        Arguments:
            ckpt_dir {str} -- Path to mode.ckpt.xxx files

        Returns:
            agent {SRLAgent} -- A SRL
        '''

        with open(ckpt_dir + 'params.json', mode='r') as f:
            attr_dict = json.load(f)

        # prevent from creating a new directory
        attr_dict['ckpt_dir'] = ckpt_dir
        agent = cls(**attr_dict)

        return agent

    def evaluate(self, I, Y, L, filename):
        '''Evaluates the predictions Y using CoNLL 2004 Shared Task script

        I, Y are zero padded to the right -- L vector carries
        the original propositon time and Y, I are scaled
        to have the same 2nd dimension as the largest proposition
        on the batch (1st dimension)(default: 250)


        Arguments:
            I {np.narray} -- 2D matrix zero padded [batch, max_time]
                representing the obsertions indices

            Y {np.narray} -- 2D matrix zero padded [batch, max_time]
                              model predictions.

            L {np.narray} -- 1D vector [batch]
                             stores the true length of the proposition

            filename {str} -- prefix for the files to used for evaluation
                            CoNLL 2004 scripts requires that the contents
                            will be saved to disk.

        Returns:
            f1 {float} -- the score

        Raises:
            AttributeError -- [description]
        '''
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

    def evaluate_dataset(self, ds_type):
        '''Evaluates the contents of ds_type using CoNLL 2004 Shared Task script

        Runs the CoNLL 2004 Shared Task script saving 3 files on target_dir
        * <ds_type>_dataset-gold.props -- propositions gold standard
        * <ds_type>_dataset-eval.props -- predicted propositions
        * <ds_type>_dataset.conll -- Overall script results

        Arguments:
            ds_type {str} -- dataset type 'train', 'valid', 'test'

        Returns:
            f1 -- evaluation score

        Raises:
            ValueError --  dataset type in ('train', 'valid', 'test')
        '''
        if ds_type in ('train', 'valid', 'test'):
            input_fn = getattr(TfStreamer, 'get_{:}'.format(ds_type))
            eval_name = '{:}_dataset'.format(ds_type)
        else:
            err = 'ds_type must be in (`valid`,`train`,`test`) got `{:}`'
            err = err.format(ds_type)
            raise ValueError(err)

        X, T, L, I = input_fn(
            self.input_labels, self.target_labels, version=self.version,
            embeddings_model=self.embeddings_model
        )

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        with tf.Session() as self.session:
            self.session.run(init_op)
            saver = tf.train.Saver()
            session_path = '{:}model.ckpt'.format(self.target_dir)
            saver.restore(self.session, session_path)
            Y, f1 = self._predict_and_eval(I, X, L, eval_name)

        return f1

    def fit(self):
        '''Trains the labeler and evaluates using CoNLL 2004 script

        Loads the training set and evaluation set
        '''
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

        with tf.Session() as self.session:
            self.session.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            saver = tf.train.Saver()
            session_path = '{:}model.ckpt'.format(self.target_dir)
            # tries to restore saved model
            if self._restore_session:
                saver.restore(self.session, session_path)
                conll_path = '{:}best-valid.conll'.format(self.target_dir)
                self.evaluator.evaluate_fromconllfile(conll_path)
                best_validation_rate = self.evaluator.f1
            else:
                best_validation_rate = -1

            # Training control variables
            step = 1
            total_loss = 0.0
            total_error = 0.0
            eps = 100
            try:
                while not (coord.should_stop() or eps < 1e-3):
                    X_batch, T_batch, L_batch, I_batch = self.session.run(self.streamer.stream)

                    loss, _, error = self.session.run(
                        [self.rnn_srl.cost, self.rnn_srl.label, self.rnn_srl.error],
                        feed_dict={self.X: X_batch, self.T: T_batch, self.L: L_batch}
                    )

                    total_loss += loss
                    total_error += error
                    if (step) % 25 == 0:

                        _, f1_train = self._predict_and_eval(I_train, X_train, L_train, 'train')


                        Y_valid, f1_valid = self._predict_and_eval(I_valid, X_valid, L_valid, 'valid')
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
                            saver.save(self.session, session_path)
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')

            finally:
                # When done, ask threads to stop
                coord.request_stop()
                coord.join(threads)

    def predict(self, X, L):
        '''Predicts the Semantic Role Labels

        X, L are zero padded to the right -- L vector carries
        the original propositon time and X is scaled
        to have the same 2nd dimension as the largest proposition
        on the batch (1st dimension)(default: 250)

        Arguments:
            I {np.narray} -- 2D matrix zero padded [batch, max_time]
                            representing the obsertions indices

            X {np.narray} -- 3D matrix zero padded [batch, max_time, features]
                             model inputs

        Returns:
            Y - {np.narray} -- 2D matrix zero padded [batch, max_time]
                              model predictions.
        '''
        Y = self.session.run(
            self.rnn_srl.predict,
            feed_dict={self.X: X, self.L: L})

        return Y

    def _predict_and_eval(self, I, X, L, evalfilename):
        Y = self.predict(X, L)
        f1 = self.evaluate(I, Y, L, evalfilename)
        return Y, f1


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
