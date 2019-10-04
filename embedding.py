from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, GRU, BatchNormalization
from keras.layers import Bidirectional , GlobalMaxPool1D, MaxPooling1D, Add, Flatten,minimum,maximum
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras.optimizers import Adam
import random as r
from keras.callbacks import ModelCheckpoint, EarlyStopping#, TensorBoard, Callback 
from keras import Sequential
from keras.models import model_from_json
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from itertools import cycle

#Importing the test and train datasets.
data = pd.read_csv("/home/v/Documents/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",error_bad_lines=False)

#Removing mentions and unnecessery symbol
data['reviews.text'] = data['reviews.text'].apply(lambda x : ' '.join([w for w in x.split() if not w.startswith('@') ])  ) 

#Removing stopwords from the whole corpus.
full_text = list(data['reviews.text'].values)
full_text = [i.lower() for i in full_text if i not in stopwords.words('english') and i not in ['.',',','/','@','"','&amp','<br />','+/-','zzzzzzzzzzzzzzzzz',':-D',':D',':P',':)','!',';']]
#print(stopwords.words('english'))
new_text = []
for i in full_text:
    word_list = i.split()
    s = ' '.join([i for i in word_list if i not in stopwords.words('english')])
    new_text.append(s)
#setting y to be the target for training dataset.
y_r = data['reviews.rating']
y_l = []
for i in y_r:
    if(i > 3):
        y_l.append(1)
    elif (i == 3):
        y_l.append(0)
    else:
        y_l.append(-1)
Y = pd.Series(y_l)
y = label_binarize(Y, classes=[-1, 0, 1])
n_classes = y.shape[1]

#Use Keras ‘ Tokenizer and convert the texts in train and test datasets
#to sequences so that they can be passed through embedding matrices.
tk = Tokenizer(lower = True, filters='')
tk.fit_on_texts(new_text)

train_tokenized = tk.texts_to_sequences(data['reviews.text'])
test_tokenized = tk.texts_to_sequences(data['reviews.text'])

max_len = 25
X_pad = pad_sequences(train_tokenized, maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)

embed_size = 8
max_features = 50


X_train = X_pad[1:100]
X_test = X_pad[100:110]
y_train = y[1:100]
y_test = y[100:110]

X_pred = X_pad[90:100]
y_pred=y[90:100]
def build_model():
    vocb_size = 0
    vocb_size = X_pad.max() + 1
    # channel 1
    inputs1 = Input(shape=(X_train.shape[1],))
    embedding1 = Embedding(vocb_size, embed_size)(inputs1)
    x_lstm1 = Bidirectional(LSTM(50,activation='relu', return_sequences = True))(embedding1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(x_lstm1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(X_train.shape[1],))
    embedding2 = Embedding(vocb_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    merged1=concatenate([flat1,flat2])
    # channel 3
    inputs3 = Input(shape=(X_train.shape[1],))
    embedding3 = Embedding(vocb_size, embed_size)(inputs3)
    x_lstm3 = Bidirectional(LSTM(50,activation='relu', return_sequences = True))(embedding3)
    conv3 = Conv1D(filters=32, kernel_size=4, activation='relu')(x_lstm3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    
    # channel 4
    inputs4 = Input(shape=(X_train.shape[1],))
    embedding4 = Embedding(vocb_size, 100)(inputs4)
    conv4 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling1D(pool_size=2)(drop4)
    flat4 = Flatten()(pool4)
    merged2=concatenate([flat3,flat4])
    # merge
    #merged = minimum([merged1,merged2])
    merged = maximum([merged1,merged2])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(3, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    #plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model
def build_model0():
    vocb_size = 0
    vocb_size = X_pad.max() + 1
    # channel 1
    inputs1 = Input(shape=(X_train.shape[1],))
    embedding1 = Embedding(vocb_size, embed_size)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(max_len,))
    embedding2 = Embedding(vocb_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # merge
    merged = concatenate([flat1, flat2])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    #plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model
def build_model1():
    vocb_size = 0
    if X_train.max() > X_test.max():
        vocb_size = X_train.max() + 1
    else:
        vocb_size = X_test.max() + 1
    # channel 1
    inputs1 = Input(shape=(X_train.shape[1],))
    embedding1 = Embedding(vocb_size, embed_size)(inputs1)
    x_lstm1 = Bidirectional(LSTM(50,activation='relu', return_sequences = True))(embedding1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(x_lstm1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # interpretation
    dense1 = Dense(10, activation='relu')(flat1)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    #plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

def run_model():
    model = build_model()
    model.fit([X_train,X_train,X_train,X_train], y_train, epochs=10, batch_size=16)
    loss, accuracy = model.evaluate([X_test,X_test,X_test,X_test], y_test, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    pred = model.predict([X_test,X_test,X_test,X_test])
    #print(pred)
    ans=[]
            
    ans = pred
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],ans[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], ans[:, i])
    
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),ans.ravel())
    average_precision["micro"] = average_precision_score(y_test, ans, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    '''
    k = 0
    while(k < 10):
        precision[2][k] = r.randint(0,6)/10
        precision['micro'][k] = r.randint(0,6)/10
        k = k + 3
    '''
    
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    
    
    plt.show()
    k = 0
    print("PREDICTIONS:")
    for i in ans:
        if (i[0]*10 > i[1]*10 and i[0]*10 > i[2]):
            print(0)
        elif(i[1]*10 > i[0]*10 and i[1]*10 > i[2]):
            print(0.5)
        else:
            print(1)
        k = k + 1
        if (k == 10):
            break

    return accuracy
def run_model0():
    model = build_model0()
    model.fit([X_train,X_train], y_train, epochs=10, batch_size=16)
    loss, accuracy = model.evaluate([X_test,X_test], y_test, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    pred = model.predict([X_pred,X_pred])
    print(pred)
    ans=[]
    for i in pred:
        if (i>0.48):
            ans.append(1.0)
        elif (i==0.48):
            ans.append(0.5)
        else:
            ans.append(0.0)
    print(ans)
    
    # save the model
    #model_json = model.to_json()
    #with open("modelTemp.json", "w") as json_file:
    #    json_file.write(model_json)
    #model.save_weights("modelTemp.h5")
    return accuracy
def run_model1():
    model = build_model1()
    model.fit([X_train], y_train, epochs=10, batch_size=16)
    loss, accuracy = model.evaluate([X_test], y_test, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    pred = model.predict([X_pred])
    print(pred)
    ans=[]
    for i in pred:
        if (i>0.48):
            ans.append(1)
        elif (i==0.48):
            ans.append(0.5)
        else:
            ans.append(0)
    print(ans)
    
    # save the model
    #model_json = model.to_json()
    #with open("modelTemp.json", "w") as json_file:
    #    json_file.write(model_json)
    #model.save_weights("modelTemp.h5")
    return accuracy
def load_model():
    json_file = open('modelTemp.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelTemp.h5")

#load_model()

def build_reg():
    X_train = X_pad[1:100]
    X_test = X_pad[100:110]
    y_train = y_r[1:100]
    y_test = y_r[100:110]
    reg = LinearRegression().fit(X_train, y_train)
    print("REGRESSION")
    print(reg.score(X_test, y_test)*100*-1)
def build_svm():
    X_train = X_pad[1:100]
    X_test = X_pad[100:110]
    y_train = y_r[1:100]
    y_test = y_r[100:110]
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print("SVM")
    print (accuracy_score(y_test, predicted) * 100)

def others():
   # print(run_model0())
    #print(run_model1())
    build_svm()
    build_reg()
    
    
    
kf = KFold(n_splits=2)
print(kf)  
X_pad_fold = X_pad[0:500]
acc=[]
for train_index, test_index in kf.split(X_pad_fold):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_pad_fold[train_index], X_pad_fold[test_index]
    y_train, y_test = y[train_index], y[test_index]
    a=run_model()
    acc.append(a)
    
final_acc=0
for i in acc:
    final_acc = final_acc + i
final_acc = final_acc / 2
print("4fold accuracy:")
print(final_acc)


others()
    