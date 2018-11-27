from time import time
import itertools, datetime
import pandas as pd
import preprocess
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda

def main(train_data_path, word2vec_path):
    df_train = pd.read_csv(train_data_path)
    word2vec = load_embeddings(word2vec_path)
    print("Preprocessing Question Columns")
    df_train, vocab, inverse_vocab = preprocess_question_cols(df_train, word2vec.vocab)
    print("Creating Embedding Matrix")
    embeddings_matrix = create_embeddings_matrix(vocab, word2vec)
    print("embedding matrix created successfully.")

    cols = ["question1", "question2", "is_duplicate"]

    max_seq_len = max(df_train.question1.map(lambda x: len(x)).max(),
                      df_train.question2.map(lambda x: len(x)).max())

    # Segregating train and validation data
    val_data_len = int(len(df_train)*0.15) # 85-15 split
    train_data_len = len(df_train) - val_data_len
    X = df_train[cols[:2]] # i.e. question1, question2
    Y = df_train[cols[2]] # i.e. label is_duplicate
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_data_len)
    X_train = {'q1': X_train.question1, 'q2': X_train.question2}
    X_val = {'q1': X_val.question1, 'q2': X_val.question2}
    Y_train, Y_val = Y_train.values, Y_val.values

    # Padding with zeros
    for df, side in itertools.product([X_train, X_val], ['q1', 'q2']):
        df[side] = pad_sequences(df[side], maxlen=max_seq_len)

    # create model - Siamese Network Architecture, shared LSTM
    hidden_units, batch_size, epochs = 128, 128, 25 
    q1_input, q2_input = Input(shape=(max_seq_len,), dtype='int32'), Input(shape=(max_seq_len,), dtype='int32')
    embeddings_layer = Embedding(len(embeddings_matrix), 300, weights=[embeddings_matrix], input_length=max_seq_len, trainable=True)
    encoded_q1, encoded_q2 = embeddings_layer(q1_input), embeddings_layer(q2_input)
    lstm = LSTM(hidden_units) 
    output_q1, output_q2 = lstm(encoded_q1), lstm(encoded_q2)
    output_distance = Lambda(function=lambda out: exp_neg_manhattan_distance(out[0], out[1]),
                             output_shape=lambda out: (out[0][0], 1))([output_q1, output_q2])
    model = Model([q1_input, q2_input], [output_distance])
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    #Callbacks
    callback = [EarlyStopping(min_delta=0.0001, patience=3), ModelCheckpoint(filepath="model.hdf5", verbose=1, save_best_only=True)]

    # Train the model
    print("Training Starts...")
    start = time()
    model = model.fit([X_train['q1'], X_train['q2']], Y_train, batch_size=batch_size, nb_epoch=epochs,
                            validation_data=([X_val['q1'], X_val['q2']], Y_val), verbose=2, callbacks=callback)
    end = time()
    print("Training successful. Finished in time {}".format(epochs, datetime.timedelta(seconds=end-start)))


    # Save the vocabulary
    save_obj(max_seq_len, "max_seq_len")
    save_obj(vocab, "vocab")

    print(model.history['acc'])
    print(model.history['val_acc'])


def exp_neg_manhattan_distance(left, right):
    ''' calculates distance between LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
    

def load_embeddings(embeddings_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        embeddings_path, binary=True)
    return model

def preprocess_question_cols(df, word2vec_vocab, vocab=None, inverse_vocab=None):
    if vocab == None:
        vocab = dict()
        inverse_vocab = ['<unk>']
    cols = ['question1', 'question2']
    stop_words = set(stopwords.words('english'))
    for i, row in df.iterrows():
        for col in cols:
            words = preprocess.process_line(row[col])
            representation = []
            for word in words:
                if word in stop_words or word not in word2vec_vocab:
                    continue
                if word not in vocab:
                    inverse_vocab.append(word)
                    vocab[word] = len(inverse_vocab)-1
                representation.append(vocab[word])
            df.set_value(i, col, representation)
    return df, vocab, inverse_vocab

def create_embeddings_matrix(vocab, word2vec, dim=300):
    matrix = np.zeros((len(vocab)+1, dim))
    for word, inverse_index in vocab.items():
        matrix[inverse_index] = word2vec.word_vec(word)
    return matrix

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__=="__main__":
    main("train_full.csv", "GoogleNews-vectors-negative300.bin")
    
    
    
    
