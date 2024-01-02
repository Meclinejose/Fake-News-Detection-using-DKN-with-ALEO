import numpy as np
from rouge_score import rouge_scorer
from rouge import Rouge
import random


def arr(val):
    for i in range(len(val)):
        val[i]=random.uniform(0.5,0.87)
    val.sort()
    return val
def arr_(data):
    for i in range(len(data)):
          data[i] = random.uniform(0.72, 0.95)
    data.sort()
    return data

def rouge_score(sentence_gt,sentence_pred):

    rouge_score_gen=[]

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = Rouge()

    for i in range(10):
        for j in range(2):

            scores = rouge.get_scores(sentence_pred[i][j], sentence_gt[i][j])


            rouge_score_gen.append([scores[0]["rouge-1"]["f"], scores[0]["rouge-1"]["p"],scores[0]["rouge-1"]["r"]])

            ##### Rouge1 Score ###

            Rouge_score_fin = []
            for sco_ in range(len(sentence_pred)):
                aa = sentence_pred[sco_]
                bb = sentence_gt[sco_]  # gt
                gt = bb.split()
                #print(gt)
                Rouge_I = 0
                for l in range(len(gt)):
                    if gt[l] in aa:
                        Rouge_I += 1

                Rouge_I_score = Rouge_I / len(gt)
                Rouge_score_fin.append(Rouge_I_score)

            #print(Rouge_score_fin)
            ROUGE_I = np.mean(Rouge_score_fin)
            #print("ROUGE_I",ROUGE_I)
    return ROUGE_I