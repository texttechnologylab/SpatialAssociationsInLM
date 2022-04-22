import os
#import dcor
import pingouin
import pandas as pd
#from scipy.stats import pearsonr, spearmanr, kendalltau

#RESULTFOLDER = "results/AllNyuResults/"
#RESULTFOLDER = "results/AllPartResults/"
RESULTFOLDER = "results/AllVerbResults/"

MODELS = ["word2vec-avg", "glove-avg", "levy-avg", "ft-avg", "bert_12layer_sent-avg"]
SCORE = ["cosim", "distcor", "kendalltau", "classify-Nearest Neighbors", "classify-Linear SVM", "classify-ffn_100_0.01_100"]

#MODELS = ["bert-base-uncased", "bert-large-uncased", "roberta-large", "electra-large-generator", "albert-xxlarge-v2"]
#SCORE = ["obj-cosim", "mask_obj-last_rep-log", "mask_room-first_rep-log", "obj-classify-Nearest Neighbors", "obj-classify-Linear SVM", "obj-classify-ffn_100_0.01_100"]
#SCORE = ["obj-cosim", "mask_obj-last_rep-log", "mask_room-first_rep-log", "obj-classify-Nearest Neighbors", "obj-classify-Linear SVM", "obj-classify100_0.01_100"]
# For Part

#MODELS = ["gpt2-large", "gpt-neo-2.7B", "gpt-j-6B"]
#SCORE = ["obj-cosim", "pred-obj-prob-1", "pred-obj-log-1", "pred-room-prob", "pred-room-log", "obj-classify-Nearest Neighbors", "obj-classify-Linear SVM", "obj-classify100_0.01_100"]

SKIP = []
cos_results = {} # room: List

def do_something(df):
    df = df.sort_values(by=["subset", "object", "room"])

    for room in sorted(set(df["room"])):
        filter_df = df[df["room"] == room]
        distcor, pval = pingouin.distance_corr(filter_df["dataset_prob"], filter_df["score"], seed=123, n_boot=1000)

        if room in cos_results:
            if pval < 0.001:
                cos_results[room].append('%.2f' % distcor + "*")
            elif pval < 0.01:
                cos_results[room].append('%.2f' % distcor + "*")
            else:
                cos_results[room].append('%.2f' % distcor)
        else:
            if pval < 0.001:
                cos_results[room] = ['%.2f' % distcor + "*"]
            elif pval < 0.01:
                cos_results[room] = ['%.2f' % distcor + "*"]
            else:
                cos_results[room] = ['%.2f' % distcor]



def do_something2(df, model):
    df = df.sort_values(by=["subset", "object", "room"])
    room = model
    filter_df = df
    # correlation = filter_df["dataset_prob"].corr(filter_df["score"])
    #distcor = dcor.distance_correlation(filter_df["dataset_prob"], filter_df["score"])
    distcor, pval = pingouin.distance_corr(filter_df["dataset_prob"], filter_df["score"], seed=123, n_boot=1000)

    if room in cos_results:
        if pval < 0.001:
            cos_results[room].append('%.2f' % distcor + "*")
        elif pval < 0.01:
            cos_results[room].append('%.2f' % distcor + "*")
        else:
            cos_results[room].append('%.2f' % distcor)
    else:
        if pval < 0.001:
            cos_results[room] = ['%.2f' % distcor + "*"]
        elif pval < 0.01:
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
                do_something2(df, "all")
            else:
                print("Could not find:", path)
                for room in cos_results.keys():
                    cos_results[room].append("-")

    print(cos_results.keys())
    for room in cos_results.keys():
        #print(cos_results[room])
        print(room)
        print(" & ".join(cos_results[room]))
