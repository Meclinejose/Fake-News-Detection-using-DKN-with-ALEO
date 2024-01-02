import Bert_Tokenization
import pandas as pd
import numpy as np
import Fea_Ext
import Label
import Proposed_ALEO_DKN.DKN
import FakeBERT.CNN
import ML.MLP
import MVAN.LSTM
import Ensemble_based_DL_model.Neural_Network
def callmain(tr):
    PRE,REC,FM,ROUGE=[],[],[],[]
    file_name='Dataset/tamil_movie_reviews_train.csv'
    data=pd.read_csv(file_name)
    data=np.array(data)
    Tamil_Txt=data[:,1]
    cls=data[:,-1]
    label=Label.get_cls(cls)
    Tok_data=Bert_Tokenization.data(Tamil_Txt)
    Feature=Fea_Ext.Ext_fea(Tok_data)

    #----------------Proposed_ALEO_DKN-----------------
    Proposed_ALEO_DKN.DKN.classify(Feature, label, Tamil_Txt, Tok_data, tr, PRE, REC, FM, ROUGE)
    #---------------Comparative Method------------
    FakeBERT.CNN.Classify(Feature, label, Tamil_Txt, Tok_data, tr, PRE, REC, FM, ROUGE)
    ML.MLP.Classify(Feature, label, Tamil_Txt, Tok_data, tr, PRE, REC, FM, ROUGE)
    MVAN.LSTM.classify(Feature, label, Tamil_Txt, Tok_data, tr, PRE, REC, FM, ROUGE)
    Ensemble_based_DL_model.Neural_Network.Classify(Feature, label, Tamil_Txt, Tok_data, tr, PRE, REC, FM, ROUGE)

    return PRE,REC,FM,ROUGE


