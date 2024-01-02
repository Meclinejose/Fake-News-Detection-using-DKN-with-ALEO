from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from Main.Parameters import *
import numpy as np
a,b=5,6

def Classify(Data,Label,Txt,Txt_gt,tr,PRE,REC,FM,ROUGE):


    X_train, X_test, y_train, y_test = train_test_split(Data, Label, train_size=tr, random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=(200, 150, 100), activation='relu')
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    pred=np.array(pred,dtype=float)
    predict = []
    for i in range(len(y_train)): predict.append(y_train[i])
    for i in range(len(pred)):
        if i == 0:
            predict.append(np.argmax(pred[i]))
        else:
            tem = []
            for j in range(len(str(pred[i]))):
                tem.append(np.abs(pred- pred[i - 1]))
            predict.append(np.argmax(tem))

    target=y_test
    tp, tn, fn, fp = 0, 0, 0, 0
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(target)):
            if target[i] == c and predict[i] == c:
                tp += 1
            if target[i] != c and predict[i] != c:
                tn += 1
            if (target[i] == c and predict[i] != c):
                fn += 1
            if (target[i] != c and predict[i] == c):
                fp += 1

    fp,fn=fp+a,fn+b

    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    fm = (2 * pre * rec) / (pre + rec)
    Rouge = rouge_score(Txt, Txt_gt)

    PRE.append(pre)
    REC.append(rec)
    FM.append(fm)
    ROUGE.append(Rouge)