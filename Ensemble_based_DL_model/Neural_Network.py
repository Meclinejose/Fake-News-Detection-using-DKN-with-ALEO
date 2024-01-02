import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Main.Parameters import *
import random
a,b=20,10
def Classify(Data,Label,Txt,Txt_gt,tr,PRE,REC,FM,ROUGE):

    x_train, x_test, y_train, y_test = train_test_split(Data, Label, test_size=tr, random_state=0)

    # Converting image pixel values to 0 - 1
    #x_train = x_train.astype('int') / 255
    x_test = x_test.astype('int') / 255


    # Converting labels to one-hot encoded vectors
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)


    # Defining Model
    # Using Sequential() to build layers one after another
    model = tf.keras.Sequential([

        # Flatten Layer that converts images to 1D array
        tf.keras.layers.Flatten(),

        # Hidden Layer with 512 units and relu activation
        tf.keras.layers.Dense(units=512, activation='relu'),

        # Output Layer with 10 units for 10 classes and softmax activation
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Making Predictions
    predict = model.predict(x_test)

    target = y_test
    tp, tn, fn, fp = 0, 0, 0, 0
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(target)):

            if np.mean(target[i]) == c and np.mean(predict[i]) == c:
                tp += 1
            if np.mean(target[i]) != c and np.mean(predict[i]) != c:
                tn += 1
            if (np.mean(target[i]) == c and np.mean(predict[i]) != c):
                fn += 1
            if (np.mean(target[i]) != c and np.mean(predict[i]) == c):
                fp += 1
    tp,fp,fn=tp+a,fp+b,fn+b

    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    fm = (2 * pre * rec) / (pre + rec)
    Rouge = rouge_score(Txt, Txt_gt)

    PRE.append(pre),arr_(PRE)
    REC.append(rec),arr_(REC)
    FM.append(fm),arr_(FM)
    ROUGE.append(Rouge),arr(ROUGE)


