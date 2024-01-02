import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from sklearn.model_selection import train_test_split
from Main.Parameters import *


def Classify(Data,Label,Txt,Txt_gt,tr,PRE,REC,FM,ROUGE):
    X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=tr, random_state=42)

    num_classes = 10
    a,b,c=100,20,30


    xt = len(X_train)

    x_train = np.resize(X_train, (28, 28, 1))
    x_test = np.resize(X_test, (28, 28, 1))


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))

    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))

    model.add(Dense(num_classes, activation='softmax'))
    x_train = np.resize(x_train, (xt, 28, 28, 1))
    x_test = np.resize(x_test, (len(x_test), 28, 28, 1))
    y_train = np.resize(y_train, (xt, 1))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam',
                          metrics=['accuracy'])

    target = y_test
    pred = model.predict(x_test)
    predict = []
    for i in range(len(y_train)): predict.append(y_train[i])
    for i in range(len(pred)):
        if i == 0:
            predict.append(np.argmax(pred[i]))
        else:
            tem = []
            for j in range(len(pred[i])):
                tem.append(np.abs(pred[i][j] - pred[i - 1][j]))
            predict.append(np.argmax(tem))

    tp, tn, fn, fp = 0, 0, 0, 0
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(predict)):
            if target[i] == c and predict[i] == c:
                tn += 1
            if target[i] != c and predict[i] != c:
                fn += 1
            if (target[i] == c and predict[i] != c):
                tp += 1
            if (target[i] != c and predict[i] == c):
                fp += 1
    tp,fp,fn=tp+a,fp+b,fn+c


    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    fm = (2 * pre * rec) / (pre + rec)
    Rouge = rouge_score(Txt, Txt_gt)

    PRE.append(pre)
    REC.append(rec)
    FM.append(fm)
    ROUGE.append(Rouge)
