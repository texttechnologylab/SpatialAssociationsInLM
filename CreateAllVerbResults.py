from BiasTester.create_w2v_results import *
from BiasTester.create_mask_model_verb_results import *
from BiasTester.create_pred_model_verb_results import *
from BiasTester.create_mask_model_results import *
from BiasTester.Datasets.Datasets import WikiHowDataset
from BiasTester.Encoder.Encoder import BertCLSEncoder, BertMASKEncoder, BertCausalEncoder
from BiasTester.DataProcessing.DataProcessor import MaskPartVerbProcessor, PredPartProcessor

from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RESULTFOLDER = "AllVerbResultsV3/"
#RESULTFOLDER = "AllMevluetResults/Verb/"


def run_mask_model(modename, dataframe, data, top_obj_to_room_map, top_objs_per_room):
    print(modename)
    savename = modename.split("/")[-1]
    clsencoder = BertCLSEncoder(modename)
    maskencoder = BertMASKEncoder(modename)

    cos_df = compute_mask_model_cos(clsencoder, dataframe, data, top_obj_to_room_map, "OBJ")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-cosim.csv", sep='t')

    cos_df = compute_mask_model_distcor(clsencoder, dataframe, data, top_obj_to_room_map, "OBJ")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-distcor.csv", sep='t')

    cos_df = compute_mask_model_distcorv2(clsencoder, dataframe, data, top_obj_to_room_map, "OBJ")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-distcor-v2.csv", sep='t')

    sentprocessor = MaskPartVerbProcessor
    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-last_rep-log.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-last_rep-log.csv", sep='t')

    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-first_rep-log.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-first_rep-log.csv", sep='t')

    classify_df = compute_mask_model_classify(clsencoder, dataframe, data, top_obj_to_room_map, top_objs_per_room, "OBJ")
    classify_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-classify.csv", sep='t')

    for epoch in [100]:
        for lr in [0.01]:
            for hidden in [100]:
                print("Run " + str(epoch) + "_" + str(lr) + "_" + str(hidden))
                classify_df = compute_mask_model_classify_ffn(clsencoder, dataframe, data, top_obj_to_room_map,
                                                          top_objs_per_room, "OBJ", hidden, epoch, lr)
                classify_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-classify-ffn_" + str(epoch) + "_" + str(lr) + "_" + str(hidden)+".csv", sep='t')


    #https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    for modelname, model in [("Nearest Neighbors", KNeighborsClassifier()), ("Linear SVM", SVC(kernel="linear", cache_size=20000))]:#, ("RBF SVM", SVC()), ("Gaussian Process", GaussianProcessClassifier())]:
        print(modelname)
        classify_df = compute_mask_model_classify_all(clsencoder, dataframe, data, top_obj_to_room_map,
                                                          top_objs_per_room, "OBJ", model)
        classify_df.to_csv(
            RESULTFOLDER + "/" + savename + "-obj-classify-" + str(modelname) + ".csv", sep='t')


def run_gen_model(modename, dataframe, data, top_obj_to_room_map, top_objs_per_room):
    print(modename)
    savename = modename.split("/")[-1]
    clsencoder = BertCLSEncoder(modename)
    predencoder = BertCausalEncoder(modename)

    cos_df = compute_pred_model_cos(clsencoder, dataframe, data, top_obj_to_room_map, "OBJ")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-cosim.csv", sep='t')

    cos_df = compute_pred_model_distcor(clsencoder, dataframe, data, top_obj_to_room_map, "OBJ")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-distcor.csv", sep='t')

    cos_df = compute_pred_model_distcorv2(clsencoder, dataframe, data, top_obj_to_room_map, "OBJ")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-distcor-v2.csv", sep='t')

    sentprocessor = PredPartProcessor
    cos_df = compute_pred_model_pred_room_prob(predencoder, dataframe, data, top_obj_to_room_map, sentprocessor)
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-room-prob.csv", sep='t')

    cos_df = compute_pred_model_pred_room_log(predencoder, dataframe, data, top_obj_to_room_map, sentprocessor, "This is usually part of")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-room-log.csv", sep='t')

    cos_df = compute_pred_model_pred_obj_prob(predencoder, dataframe, data, top_obj_to_room_map, "I usually {room} this")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-obj-prob.csv", sep='t')

    cos_df = compute_pred_model_pred_obj_log(predencoder, dataframe, data, top_obj_to_room_map, "I usually {room} this")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-obj-log.csv", sep='t')

    cos_df = compute_pred_model_pred_obj_prob(predencoder, dataframe, data, top_obj_to_room_map, "I usually {room} this", rep=-1)
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-obj-prob-1.csv", sep='t')

    cos_df = compute_pred_model_pred_obj_log(predencoder, dataframe, data, top_obj_to_room_map, "I usually {room} this", rep=-1)
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-obj-log-1.csv", sep='t')

    classify_df = compute_pred_model_classify(clsencoder, dataframe, data, top_obj_to_room_map, top_objs_per_room, "OBJ")
    classify_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-classify.csv", sep='t')

    for epoch in [100]:
        for lr in [0.01]:
            for hidden in [100]:
                print("Run " + str(epoch) + "_" + str(lr) + "_" + str(hidden))
                classify_df = compute_pred_model_classify_ffn(clsencoder, dataframe, data, top_obj_to_room_map,
                                                          top_objs_per_room, "OBJ", hidden, lr, epoch)
                classify_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-classify" + str(epoch) + "_" + str(lr) + "_" + str(hidden)+".csv", sep='t')


    #https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    for modelname, model in [("Nearest Neighbors", KNeighborsClassifier()), ("Linear SVM", SVC(kernel="linear", cache_size=20000))]:#, ("RBF SVM", SVC()), ("Gaussian Process", GaussianProcessClassifier())]:
        print(modelname)
        classify_df = compute_pred_model_classify_all(clsencoder, dataframe, data, top_obj_to_room_map,
                                                          top_objs_per_room, "OBJ", model)
        classify_df.to_csv(
            RESULTFOLDER + "/" + savename + "-obj-classify-" + str(modelname) + ".csv", sep='t')

def run_static_model(modelpath, output, dataframe, data, top_obj_to_room_map, top_objs_per_room, bin, no_header):
    print(output)
    rep = "avg"
    model = KeyedVectors.load_word2vec_format(datapath(modelpath), binary=bin, no_header=no_header)


    cos_df = compute_w2v_cos(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-cosim.csv", sep='t')

    cos_df = compute_w2v_distcor(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-distcor.csv", sep='t')

    classify_df = compute_w2v_classify(dataframe, data, top_obj_to_room_map, top_objs_per_room, model, rep)
    classify_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-classify.csv", sep='t')

    cos_df = compute_w2v_pearson(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-pearson.csv", sep='t')


    cos_df = compute_w2v_spearman(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-spearman.csv", sep='t')

    cos_df = compute_w2v_kendalltau(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-kendalltau.csv", sep='t')


    for epoch in [100]: #[25, 50, 100]:
        for lr in [0.01]:
            for hidden in [100]:#, 200, 300]:
                print("Run " + str(epoch) + "_" + str(lr) + "_" + str(hidden))
                classify_df = compute_w2v_classify_ffn(dataframe, data, top_obj_to_room_map,
                                                          top_objs_per_room, model, rep, hidden, lr, epoch)
                classify_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-classify-ffn_" + str(epoch) + "_" + str(lr) + "_" + str(hidden)+".csv", sep='t')


    #https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    for modelname, classifier in [("Nearest Neighbors", KNeighborsClassifier()), ("Linear SVM", SVC(kernel="linear", cache_size=20000))]:#, ("RBF SVM", SVC()), ("Gaussian Process", GaussianProcessClassifier())]:
        print(modelname)
        classify_df = compute_w2v_classify_all(dataframe, data, top_obj_to_room_map,
                                                          top_objs_per_room, model, rep, classifier)
        classify_df.to_csv(
            RESULTFOLDER + "/" + output + "-" + rep + "-classify-" + str(modelname) + ".csv", sep='t')


def run():
    data = WikiHowDataset("data/howtokb_modified.csv")

    verbs = ["read", "wash with", "play", "eat", "wear", "listen to"]
    top_objs_per_verb = {}  # {room: obj}
    top_obj_to_verb_map = {}  # {obj: room}
    all_objects = []  # [obj]
    for verb in verbs:
        top_objs = data.get_most_important_obj_for_index_n(verb, 10)
        top_objs_per_verb[verb] = top_objs
        all_objects.extend(top_objs)
        for obj in top_objs:
            top_obj_to_verb_map[obj] = verb


    empty_df = pd.DataFrame(index=all_objects, columns=verbs)


    run_static_model("E:/Corpora/Embeddings/Word2Vec/GoogleNews-vectors-negative300.bin", "word2vec", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, True, False)
    run_static_model("E:/Corpora/Embeddings/Glove/glove.840B.300d.txt", "glove", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, False, True)
    run_static_model("E:/Corpora/Embeddings/fastText/crawl-300d-2M-subword.vec", "ft", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, False, False)
    run_static_model("E:/Corpora/Embeddings/Levy/deps.words", "levy", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, False, True)
    run_static_model("E:/Corpora/Embeddings/StaticBert/bert_12layer_sent.vec", "bert_12layer_sent", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, False, False)

    run_static_model("D:/Corpora/Embeddings/Bert/bert_12layer_sent.vec", "bert_12layer_sent", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, False, False)
    run_static_model("D:/Corpora/Embeddings/Bert/bert_12layer_para.vec", "bert_12layer_para", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, False, False)
    run_static_model("D:/Corpora/Embeddings/Bert/bert_24layer_sent.vec", "bert_24layer_sent", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, False, False)
    run_static_model("D:/Corpora/Embeddings/Bert/GPT2_12layer_sent.vec", "GPT2_12layer_sent", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, False, False)
    run_static_model("D:/Corpora/Embeddings/Bert/roberta_12layer_sent.vec", "roberta_12layer_sent", empty_df, data, top_obj_to_verb_map, top_objs_per_verb, False, False)


    run_mask_model('bert-base-uncased', empty_df, data, top_obj_to_verb_map, top_objs_per_verb)
    run_mask_model('bert-large-uncased', empty_df, data, top_obj_to_verb_map, top_objs_per_verb)
    run_mask_model('bert-large-uncased-whole-word-masking', empty_df, data, top_obj_to_verb_map, top_objs_per_verb)
    run_mask_model("albert-xxlarge-v2", empty_df, data, top_obj_to_verb_map, top_objs_per_verb)
    run_mask_model("roberta-large", empty_df, data, top_obj_to_verb_map, top_objs_per_verb)
    run_mask_model("google/electra-large-generator", empty_df, data, top_obj_to_verb_map, top_objs_per_verb)
    run_mask_model("google/electra-large-discriminator", empty_df, data, top_obj_to_verb_map, top_objs_per_verb)


    run_gen_model("gpt2-large", empty_df, data, top_obj_to_verb_map, top_objs_per_verb)
    run_gen_model("EleutherAI/gpt-neo-2.7B", empty_df, data, top_obj_to_verb_map, top_objs_per_verb)
    run_gen_model("xlnet-large-cased", empty_df, data, top_obj_to_verb_map, top_objs_per_verb)
    run_gen_model("EleutherAI/gpt-j-6B", empty_df, data, top_obj_to_verb_map, top_objs_per_verb)

    # Static Models

if __name__ == "__main__":
    run()

