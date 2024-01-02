from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

def lin_similarity(words):
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    n = []
    for i in range(len(words)):
        try:
            n1 = words[i]
            n2 = words[i + 1]
            l_sim = wn.synset(n1 + '.n.01').lin_similarity(wn.synset(n2 + '.n.01'), semcor_ic)
            n.append(l_sim)
        except:
            n.append(0)
    return np.mean(n)

# TF-IDF
def TF_IDF(sentence):
    b = []
    for i in range(len(sentence)):
        c = str(sentence[i])
        b.append(str(c))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(b)
    x = X.toarray()
    x = np.mean(x, axis=1)
    return np.mean(x)

# Numerical Value
def Numerical_value(sentence):
    NW_count = 0
    for char in sentence:
        if char.isdigit():
            NW_count += 1
    return NW_count

# Hastag
def Get_Hashtag(text):
    hash='#'
    hash_count = 0
    for char in text:
        if hash in char:
            hash_count += 1
    return hash_count

# Punctuation
def get_punctuation(sentence):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    punct_count = 0
    for char in sentence:
        if char in punctuations:
            punct_count += 1
    return punct_count

# Numerical word
def get_numerical_word(sentence):
    numerals = ['ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 'ஆறு', 'நூறு', 'ஏழு', 'ஒன்பது', 'ஒரு', 'இரு','அறு', 'எழு']
    word_count = 0
    for char in sentence:
        if char in numerals:
            word_count += 1
    return word_count

#Word2vector
def word_2_Vector(data):


    # Create a Vectorizer Object
    vectorizer = CountVectorizer()

    vectorizer.fit(data)

    # Encode the Document
    vector = vectorizer.transform(data)

    # Summarizing the Encoded Texts

    return np.mean(vector.toarray())



def fea(data):
    F1,F2,F3,F4,F5,F6,F7=[],[],[],[],[],[],[]
    for i in range(len(data)):
        print("i :",i)
        f1=word_2_Vector(data[i])
        F1.append(f1)
        f2=Numerical_value(data[i])
        F2.append(f2)
        f3=Get_Hashtag(data[i])
        F3.append(f3)
        f4=get_punctuation(data[i])
        F4.append(f4)
        f5=get_numerical_word(data[i])
        F5.append(f5)
        f6=lin_similarity(data[i])
        F6.append(f6)
        f7=TF_IDF(data[i])
        F7.append(f7)
    Fea=np.column_stack((F1,F2,F3,F4,F5,F6,F7))
    np.savetxt("Processed/Features.csv",Fea,delimiter=',',fmt="%s")


def Ext_fea(data):

    #fea(data)

    Feature=pd.read_csv("Processed/Features.csv",header=None)
    Feature=np.array(Feature)

    return Feature