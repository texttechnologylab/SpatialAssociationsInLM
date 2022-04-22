import os
#import dcor
import pingouin
import pandas as pd
from small_experiments.AskGoogleNGram import runQuery
#from scipy.stats import pearsonr, spearmanr, kendalltau


GRAPHFOLDERPATH = "AllNyuResults/Graphics/"
#RESULTFOLDER = "AllMevluetResults/Verb/"
RESULTFOLDER = "AllNyuResultsV2/"
#RESULTFOLDER = "AllPartResultsV2/"
#RESULTFOLDER = "AllVerbResultsV2/"

#MODELS = ["word2vec-avg", "glove-avg", "levy-avg", "ft-avg", "bert_12layer_sent-avg"]
#SCORE = ["cosim", "distcor", "kendalltau", "classify-Nearest Neighbors", "classify-Linear SVM", "classify-ffn_100_0.01_100"]

MODELS = ["bert-base-uncased", "bert-large-uncased", "roberta-large", "electra-large-generator", "albert-xxlarge-v2"]
SCORE = ["obj-cosim", "mask_obj-last_rep-log", "mask_room-first_rep-log", "obj-classify-Nearest Neighbors", "obj-classify-Linear SVM", "obj-classify-ffn_100_0.01_100"]

#MODELS = ["gpt2-large", "gpt-neo-2.7B", "gpt-j-6B"]
#SCORE = ["obj-cosim", "pred-obj-prob-1", "pred-obj-log-1", "pred-room-prob", "pred-room-log", "obj-classify-Nearest Neighbors", "obj-classify-Linear SVM", "obj-classify100_0.01_100"]


#MODELS = ["bert_12layer_sent-max", "bert_12layer_para-max","bert_24layer_sent-max", "roberta_12layer_sent-max", "gpt2_12layer_sent-max"
#MODELS = ["gpt2-large"]
#SCORE = ["mask_room-log-true_true", "mask_room-log-false_true", "mask_room-log-true_false", "mask_room-log-false_false"]
#SKIP = ["electra-large-generator-obj-cosim.csv", "electra-large-discriminator-obj-cosim.csv", "albert-xxlarge-v2-obj-cosim.csv"]
#SCORE = ["mask_room-first_rep-log", "mask_room-first_rep-log-i", "mask_room-first_rep-log-u", "mask_room-first_rep-log-u-i"]
#SCORE = ["mask_room-last_rep-log", "mask_room-last_rep-log-i", "mask_room-last_rep-log-u", "mask_room-last_rep-log-u-i"]
#SCORE = ["mask_obj-first_rep-log", "mask_obj-first_rep-log-i", "mask_obj-first_rep-log-u", "mask_obj-first_rep-log-u-i"]
#SCORE = ["mask_obj-last_rep-log", "mask_obj-last_rep-log-i", "mask_obj-last_rep-log-u", "mask_obj-last_rep-log-u-i"]
#SCORE = ["mask_room-first_rep-log", "mask_room-last_rep-log", "mask_obj-first_rep-log", "mask_obj-last_rep-log"]
#SCORE = ["obj-classify10_0.01_50", "obj-classify10_0.01_100","obj-classify10_0.001_50", "obj-classify10_0.001_100",
#         "obj-classify25_0.01_50", "obj-classify25_0.01_100","obj-classify25_0.001_50", "obj-classify25_0.001_100"]
#MODELS = ["bert-large-uncased-whole-word-masking-finetuned-squad", "roberta-large-squad2", "xlm-roberta-large-ner-hrl"]
#SCORE = ["obj-cosim", "obj-distcor", "mask_room-first_rep-log", "obj-classify"]

SKIP = []
cos_results = {} # room: List

saved_results = {}

def do_something(df):
    df = df.sort_values(by=["subset", "object", "room"])

    for room in sorted(set(df["room"])):
        print(room)
        filter_df = df[df["room"] == room]
        filter_df = filter_df[filter_df["subset"] == room]
        #print(filter_df)
        objects = list(filter_df["object"])

        if room in saved_results:
            query_df = saved_results[room]
        else:
            query = ", ".join(objects)
            query += " -caseInsensitive"
            print(query)
            query_df = runQuery(query)
            query_df = query_df.mean(axis=0)
            query_df = query_df.drop("year")
            saved_results[room] = query_df
        #print(filter_df["dataset_prob"])
        #print(query_df)
        #distcor = dcor.distance_correlation(filter_df["dataset_prob"], filter_df["score"])
        distcor, pval = pingouin.distance_corr(filter_df["score"], query_df, seed=123, n_boot=1000)
        print(distcor, pval)
        #pearsonrv = pearsonr(filter_df["dataset_prob"], filter_df["score"])[0]
        #spearmanrv = spearmanr(filter_df["dataset_prob"], filter_df["score"])[0]
        #kendalltauv = kendalltau(filter_df["dataset_prob"], filter_df["score"])[0]
        #print(pval)
        if room in cos_results:
            if pval < 0.05:
                cos_results[room].append('%.2f' % distcor + "*")
            elif pval < 0.1:
                cos_results[room].append('%.2f' % distcor + "*")
            else:
                cos_results[room].append('%.2f' % distcor)
        else:
            if pval < 0.05:
                cos_results[room] = ['%.2f' % distcor + "*"]
            elif pval < 0.1:
                cos_results[room] = ['%.2f' % distcor + "*"]
            else:
                cos_results[room] = ['%.2f' % distcor]


if __name__ == "__main__":

    for model in MODELS:
        for score in SCORE:
            #path = os.path.join(RESULTFOLDER, model+"-"+score+".csv")
            path = os.path.join(RESULTFOLDER, model+"-"+score+".csv")
            print(path)
            if model+"-"+score+".csv" in SKIP:
                continue
            if os.path.isfile(path):
                df = pd.read_csv(path, sep='t', index_col=0)
                do_something(df)
            else:
                print("Could not find:", path)
                for room in cos_results.keys():
                    cos_results[room].append("-")

    print(cos_results.keys())
    for room in cos_results.keys():
        #print(cos_results[room])
        print(room)
        print(" & ".join(cos_results[room]))
