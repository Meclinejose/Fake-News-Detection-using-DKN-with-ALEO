from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Multiply
from keras.models import Model
from Main.Parameters import *
import Proposed_ALEO_DKN.ALEA
def create_dkn_model(input_shape):
    input_layer = Input(shape=input_shape)

    # Kronecker factored layer (Convolution)
    kronecker_conv1 = Conv2D(32, (3, 3), padding='same')(input_layer)
    kronecker_conv2 = Conv2D(32, (3, 3), padding='same')(input_layer)
    kronecker_layer = Multiply()([kronecker_conv1, kronecker_conv2])

    # Flatten the Kronecker factored layer
    flattened = Flatten()(kronecker_layer)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    output_layer = Dense(1, activation='softmax')(fc1)  # Adjust the number of output classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def classify(Data,Label,Txt,Txt_gt,tr,PRE,REC,FM,ROUGE):
    x_train, x_test, y_train, y_test = train_test_split(Data, Label, train_size=tr,
                                                        random_state=0)  # 70% for train and 30% for test

    Model=create_dkn_model(input_shape=(10,10,10))
    Model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    x_train=np.array(x_train).astype('int')
    x_test=np.array(x_test).astype('int')
    y_train=np.array(y_train).astype('int')
    x_train=np.resize(x_train,(len(x_train),10,10,10))
    x_test=np.resize(x_test,(len(x_test),10,10,10))
    Model.fit(x_train,y_train)
    pred=Model.predict(x_test)
    Weight=Model.get_weights()
    Weight[0]= Proposed_ALEO_DKN.ALEA.Algm() + Weight[0]
    Model.set_weights(Weight)

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
    target=y_test
    tp, tn, fn, fp = 0, 0, 0, 0
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(target)):
            if target[i] == c and predict[i] == c:
                tn += 1
            if target[i] != c and predict[i] != c:
                fn += 1
            if (target[i] == c and predict[i] != c):
                tp += 1
            if (target[i] != c and predict[i] == c):
                fp += 1


    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    fm = (2 * pre * rec) / (pre + rec)
    Rouge = rouge_score(Txt, Txt_gt)

    PRE.append(pre)
    REC.append(rec)
    FM.append(fm)
    ROUGE.append(Rouge)

