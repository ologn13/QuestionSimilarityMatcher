from keras.models import load_model
import numpy as np
import preprocess
import pickle
import keras.backend as K

MODEL_PATH = "model.hdf5"
MAX_SEQ_LEN = "max_seq_len"
VOCAB = "vocab"

def exp_neg_manhattan_distance(left, right):
    ''' calculates distance between LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class TrainedModel:

    model = load_model(MODEL_PATH, custom_objects={
        "exp_neg_manhattan_distance": exp_neg_manhattan_distance})

    max_seq_len = load_obj(MAX_SEQ_LEN)
    vocab = load_obj(VOCAB)

    @staticmethod
    def gen_representation(sent_input):
        representation = [0 for i in range(0, TrainedModel.max_seq_len)]
        i = 0
        for word in sent_input:
            if word in TrainedModel.vocab:
                representation[i] = TrainedModel.vocab[word]
                i += 1
        return np.asarray(representation)

    @staticmethod
    def predict(sent1, sent2):
        words_sent1 = preprocess.process_line(sent1)[:TrainedModel.max_seq_len]
        words_sent2 = preprocess.process_line(sent2)[:TrainedModel.max_seq_len]
        sent1_input, sent2_input = TrainedModel.gen_representation(sent1), TrainedModel.gen_representation(sent2)
        res = TrainedModel.model.predict([[sent1_input], [sent2_input]])
        print(" {}, {} \n Score: {}".format(sent1, sent2, res))
        return "Yes" if res>=0.5 else "No"
