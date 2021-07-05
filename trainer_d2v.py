"""Training-related module.
"""
from callbacks import F1score
from utils import NERSequence_D2V
import numpy as np
from gensim.models import doc2vec
import sys


class Trainer(object):
    """A trainer that train the model.

    Attributes:
        _model: Model.
        _preprocessor: Transformer. Preprocessing data for feature extraction.
    """

    def __init__(self, model, d2v_model, d2v_dim, doc_partition_indices=None, preprocessor=None):
        self._model = model
        self._preprocessor = preprocessor
        self.d2v_model = d2v_model
        self.d2v_dim = d2v_dim
        self.doc_partition_indices = doc_partition_indices

    def train(self, x_train, y_train, x_valid=None, y_valid=None,
              epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            x_train: list of training data.
            y_train: list of training target (label) data.
            x_valid: list of validation data.
            y_valid: list of validation target (label) data.
            batch_size: Integer.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch). `shuffle` will default to True.
        """
        d2v_vectors = np.zeros((len(x_train), self.d2v_dim)).astype('float64')
        if self.doc_partition_indices is None:
            for i,sent in enumerate(x_train):
                sys.stdout.write('\r')
                sys.stdout.write(str((i*100)/len(x_train))[:5]+'% d2v vector inferencing done    ')
                sys.stdout.flush()
                d2v_vec = self.d2v_model.infer_vector(sent,steps=15)
                d2v_vectors[i] = d2v_vec
            print()

        else:
            initial_ind = 0
            for end_ind in self.doc_partition_indices:
                sys.stdout.write('\r')
                sys.stdout.write(str((end_ind*100)/len(self.doc_partition_indices))[:5]+'% d2v vector inferencing done    ')
                sys.stdout.flush()
                sentencified_doc = x_train[initial_ind:end_ind]
                tokenized_doc = []
                for sent in sentencified_doc:
                    tokenized_doc.extend(sent)
                d2v_vec = self.d2v_model.infer_vector(tokenized_doc,steps=15)
                for i,sent in enumerate(sentencified_doc):
                    d2v_vectors[i+initial_ind] = d2v_vec
                initial_ind = end_ind
            print()



        train_seq = NERSequence_D2V(x_train, y_train, d2v_vectors=d2v_vectors, batch_size=batch_size, preprocess=self._preprocessor.transform)

        if x_valid and y_valid:
            valid_seq = NERSequence_D2V(x_valid, y_valid, batch_size, self._preprocessor.transform)
            f1 = F1score(valid_seq, preprocessor=self._preprocessor)
            callbacks = [f1] + callbacks if callbacks else [f1]

        self._model.fit_generator(generator=train_seq,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=verbose,
                                  shuffle=shuffle)
