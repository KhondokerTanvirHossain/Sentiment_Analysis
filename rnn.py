import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from nltk import TweetTokenizer
from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, GRU, BatchNormalization
from keras.layers import Bidirectional, CuDNNGRU, CuDNNLSTM, BatchNormalization, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
#from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
#from keras import backend as K
#from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping#, TensorBoard, Callback 
from keras import Sequential

#Importing the test and train datasets.
train = pd.read_csv("/home/v/Documents/SENTIMENT_DATA/train.csv",sep=';')
test = pd.read_csv("/home/v/Documents/SENTIMENT_DATA/test.csv",sep=';',quoting=csv.QUOTE_NONE)
del test['Unnamed: 1']

#Removing mentions and unnecessery symbol
train['tweet'] = train['tweet'].apply(lambda x : ' '.join([w for w in x.split() if not w.startswith('@') ])  ) 
test['tweet'] = test['tweet'].apply(lambda x : ' '.join([w for w in x.split() if not w.startswith('@') ])  )

#Removing stopwords from the whole corpus.
full_text = list(train['tweet'].values) + list(test['tweet'].values)
full_text = [i.lower() for i in full_text if i not in stopwords.words('english') and i not in ['.',',','/','@','"','&amp','<br />','+/-','zzzzzzzzzzzzzzzzz',':-D',':D',':P',':)','!',';']]

#setting y to be the target for training dataset.
y = train['sentiment']

#Use Keras ‘ Tokenizer and convert the texts in train and test datasets
#to sequences so that they can be passed through embedding matrices.
tk = Tokenizer(lower = True, filters='')
tk.fit_on_texts(full_text)

train_tokenized = tk.texts_to_sequences(train['tweet'])
test_tokenized = tk.texts_to_sequences(test['tweet'])

#max_len = 50
max_len = 5
X_train = pad_sequences(train_tokenized, maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)

#Preparing the embedding matrix.
embedding_path = "/home/v/Documents/SENTIMENT_DATA/numberbatch-en-17.02.txt"
#embed_size = "1984681 300"
embed_size = 300
max_features = 30000

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

#embedding matrix
word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    if embedding_vector is not None: print(embedding_vector)

#make a deep recurrent neural network that contains an 
#Embedding layer, 
#Bidirectional CuDNN LSTM and 
#GRUs (which are Nvidia’s fastened versions of LSTM and GRU networks), 
#Convolutional layers followed by Batch Normalization and dropout layers. 
#Here we are building a model and fitting it and saving the best model to ‘saved_model.hdf5’ after every epoch if the validation accuracy is increased
def build_model(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):
    file_path = "saved_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)
    
    inp = Input(shape = (max_len,))
    x = Embedding(30001, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)
    
    x_gru = Bidirectional(GRU(units, return_sequences = True))(x1)
    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = GlobalAveragePooling1D()(x3)
    max_pool3_gru = GlobalMaxPooling1D()(x3)
    
    x_lstm = Bidirectional(LSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = GlobalAveragePooling1D()(x3)
    max_pool3_lstm = GlobalMaxPooling1D()(x3)
    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, y, batch_size = 128, epochs = 10, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model
model = build_model(lr = 1e-3, lr_d = 1e-10, units = 128, spatial_dr = 0.5, kernel_size1=2, kernel_size2=2, dense_units=64, dr=0.2, conv_size=32)
