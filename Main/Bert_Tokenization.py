from transformers import BertTokenizer
import csv
import pandas as pd
import numpy as np

def preprocess(datas):
    Tokenized_data = []

    for i in range(len(datas)):
        print("i :",i)
        tz = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        tz.convert_tokens_to_ids(["characteristically"])

        Tokenized_data.append(tz.tokenize(datas[i]))


        # # writing to csv file
        with open("Processed\\Tokenized_data.csv", 'w',encoding='utf-8',newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the data rows
            csvwriter.writerows(Tokenized_data)


    return Tokenized_data


def data(datas):
    #preprocess(datas)

    #The data is processed and stored in the path given below
    Tok_data=[]
    # opening the CSV file
    with open("Processed\\Tokenized_data.csv", mode='r',encoding='utf-8')as file:
        # reading the CSV file
        csvFile = csv.reader(file)

        # displaying the contents of the CSV file
        for lines in csvFile:

            Tok_data.append(lines)

    return Tok_data




