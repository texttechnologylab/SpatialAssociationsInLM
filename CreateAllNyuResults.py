from BiasTester.create_w2v_results import *
from BiasTester.create_mask_model_results import *
from BiasTester.create_pred_model_results import *
from BiasTester.Datasets.Datasets import NyuV2ObjDataset
from BiasTester.Encoder.Encoder import BertCLSEncoder, BertMASKEncoder, BertCausalEncoder
from BiasTester.DataProcessing.DataProcessor import MaskObjProcessor, MaskObjProcessorI, PredObjProcessor
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd

RESULTFOLDER = "results/AllNyuResults"


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


    sentprocessor = MaskObjProcessor
    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-last_rep-log-u.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-last_rep-log-u.csv", sep='t')

    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-first_rep-log-u.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-first_rep-log-u.csv", sep='t')

    sentprocessor = MaskObjProcessor
    sentprocessor.TEMPLATE = "{article} {obj} is in the {room}."
    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-last_rep-log.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-last_rep-log.csv", sep='t')

    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-first_rep-log.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-first_rep-log.csv", sep='t')


    sentprocessor = MaskObjProcessorI
    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-last_rep-log-u-i.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-last_rep-log-u-i.csv", sep='t')

    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-first_rep-log-u-i.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-first_rep-log-u-i.csv", sep='t')

    sentprocessor = MaskObjProcessorI
    sentprocessor.TEMPLATE = "In the {room} is {article} {obj}."
    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-last_rep-log-i.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-last_rep-log-i.csv", sep='t')

    log_obj_df = compute_mask_model_mask_instance_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_obj-first_rep-log-i.csv", sep='t')
    log_room_df = compute_mask_model_mask_concept_log(maskencoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=0)
    log_room_df.to_csv(RESULTFOLDER + "/" + savename + "-mask_room-first_rep-log-i.csv", sep='t')


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


def run_gen_model(modename, dataframe  , data, top_obj_to_room_map, top_objs_per_room):
    print(modename)
    savename = modename.split("/")[-1]
    clsencoder = BertCLSEncoder(modename)
    predencoder = BertCausalEncoder(modename)
    sentprocessor = PredObjProcessor

    cos_df = compute_pred_model_cos(clsencoder, dataframe, data, top_obj_to_room_map, "OBJ")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-cosim.csv", sep='t')

    cos_df = compute_pred_model_distcor(clsencoder, dataframe, data, top_obj_to_room_map, "OBJ")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-distcor.csv", sep='t')

    cos_df = compute_pred_model_distcorv2(clsencoder, dataframe, data, top_obj_to_room_map, "OBJ")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-distcor-v2.csv", sep='t')


    cos_df = compute_pred_model_pred_room_prob(predencoder, dataframe, data, top_obj_to_room_map, sentprocessor)
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-room-prob.csv", sep='t')

    cos_df = compute_pred_model_pred_room_log(predencoder, dataframe, data, top_obj_to_room_map, sentprocessor, "This is usually in the")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-room-log.csv", sep='t')

    cos_df = compute_pred_model_pred_obj_prob(predencoder, dataframe, data, top_obj_to_room_map, "In the {room} is usually")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-obj-prob.csv", sep='t')

    cos_df = compute_pred_model_pred_obj_log(predencoder, dataframe, data, top_obj_to_room_map, "In the {room} is usually")
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-obj-log.csv", sep='t')
    
    cos_df = compute_pred_model_pred_room_prob(predencoder, dataframe, data, top_obj_to_room_map, sentprocessor, pred_rep=-1)
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-room-prob-1.csv", sep='t')

    cos_df = compute_pred_model_pred_room_log(predencoder, dataframe, data, top_obj_to_room_map, sentprocessor, "This is usually in the", pred_rep=-1)
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-room-log-1.csv", sep='t')

    cos_df = compute_pred_model_pred_obj_prob(predencoder, dataframe, data, top_obj_to_room_map, "In the {room} is usually", pred_rep=-1)
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-obj-prob-1.csv", sep='t')

    cos_df = compute_pred_model_pred_obj_log(predencoder, dataframe, data, top_obj_to_room_map, "In the {room} is usually", pred_rep=-1)
    cos_df.to_csv(RESULTFOLDER + "/" + savename + "-pred-obj-log-1.csv", sep='t')


    classify_df = compute_pred_model_classify(clsencoder, dataframe, data, top_obj_to_room_map, top_objs_per_room, "OBJ")
    classify_df.to_csv(RESULTFOLDER + "/" + savename + "-obj-classify.csv", sep='t')

    for epoch in [100]:
        for lr in [0.01]:
            for hidden in [100]:
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
    rep = "avg"
    model = KeyedVectors.load_word2vec_format(datapath(modelpath), binary=bin, no_header=no_header)

    cos_df = compute_w2v_cos(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-cosim.csv", sep='t')

    cos_df = compute_w2v_distcor(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-distcor.csv", sep='t')

    cos_df = compute_w2v_mahalanobis(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-mahalanobis.csv", sep='t')

    cos_df = compute_w2v_minkowski(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-minkowski.csv", sep='t')

    cos_df = compute_w2v_pearson(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-pearson.csv", sep='t')

    cos_df = compute_w2v_spearman(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-spearman.csv", sep='t')

    cos_df = compute_w2v_kendalltau(dataframe, data, top_obj_to_room_map, model, rep)
    cos_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-kendalltau.csv", sep='t')

    for epoch in [100]: #[25, 50, 100]:
        for lr in [0.01]:
            for hidden in [100]:#, 200, 300]:
                classify_df = compute_w2v_classify_ffn(dataframe, data, top_obj_to_room_map,
                                                          top_objs_per_room, model, rep, hidden, lr, epoch)
                classify_df.to_csv(RESULTFOLDER + "/" + output + "-" + rep + "-classify-ffn_" + str(epoch) + "_" + str(lr) + "_" + str(hidden)+".csv", sep='t')


    #https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    for modelname, classifier in [("Nearest Neighbors", KNeighborsClassifier()), ("Linear SVM", SVC(kernel="linear", cache_size=20000))]:#, ("RBF SVM", SVC()), ("Gaussian Process", GaussianProcessClassifier())]:
        classify_df = compute_w2v_classify_all(dataframe, data, top_obj_to_room_map,
                                                          top_objs_per_room, model, rep, classifier)
        classify_df.to_csv(
            RESULTFOLDER + "/" + output + "-" + rep + "-classify-" + str(modelname) + ".csv", sep='t')


def run():
    data = NyuV2ObjDataset("1_DataExtraction/nyuv2.csv")

    rooms = data.get_indeces()

    top_objs_per_room = {}  # {room: obj}
    top_obj_to_room_map = {}  # {obj: room}
    all_objects = []  # [obj]
    for room in rooms:
        top_objs = data.get_most_important_obj_for_index_n(room, 10)
        top_objs_per_room[room] = top_objs
        all_objects.extend(top_objs)
        for obj in top_objs:
            top_obj_to_room_map[obj] = room

    empty_df = pd.DataFrame(index=all_objects, columns=rooms)

    # Run Static Models
    run_static_model("data/embeddings/Word2Vec/GoogleNews-vectors-negative300.bin", "word2vec", empty_df, data, top_obj_to_room_map, top_objs_per_room, True, False)
    run_static_model("data/embeddings/Glove/glove.840B.300d.txt", "glove", empty_df, data, top_obj_to_room_map, top_objs_per_room, False, True)
    run_static_model("data/embeddings/fastText/crawl-300d-2M-subword.vec", "ft", empty_df, data, top_obj_to_room_map, top_objs_per_room, False, False)
    run_static_model("data/embeddings/Levy/deps.words", "levy", empty_df, data, top_obj_to_room_map, top_objs_per_room, False, True)

    # Run Static BERT Models
    run_static_model("data/embeddings/Bert/bert_12layer_sent.vec", "bert_12layer_sent", empty_df, data, top_obj_to_room_map, top_objs_per_room, False, False)
    run_static_model("data/embeddings/Bert/bert_12layer_para.vec", "bert_12layer_para", empty_df, data, top_obj_to_room_map, top_objs_per_room, False, False)
    run_static_model("data/embeddings/Bert/bert_24layer_sent.vec", "bert_24layer_sent", empty_df, data, top_obj_to_room_map, top_objs_per_room, False, False)
    run_static_model("data/embeddings/Bert/GPT2_12layer_sent.vec", "GPT2_12layer_sent", empty_df, data, top_obj_to_room_map, top_objs_per_room, False, False)
    run_static_model("data/embeddings/Bert/roberta_12layer_sent.vec", "roberta_12layer_sent", empty_df, data, top_obj_to_room_map, top_objs_per_room, False, False)

    # Run MASK Models
    run_mask_model('bert-base-uncased', empty_df, data, top_obj_to_room_map, top_objs_per_room)
    run_mask_model('bert-large-uncased', empty_df, data, top_obj_to_room_map, top_objs_per_room)
    run_mask_model('bert-large-uncased-whole-word-masking', empty_df, data, top_obj_to_room_map, top_objs_per_room)
    run_mask_model("albert-xxlarge-v2", empty_df, data, top_obj_to_room_map, top_objs_per_room)
    run_mask_model("roberta-large", empty_df, data, top_obj_to_room_map, top_objs_per_room)
    run_mask_model("google/electra-large-generator", empty_df, data, top_obj_to_room_map, top_objs_per_room)
    run_mask_model("google/electra-large-discriminator", empty_df, data, top_obj_to_room_map, top_objs_per_room)

    # Run GEN Models
    run_gen_model("gpt2-large", empty_df, data, top_obj_to_room_map, top_objs_per_room)
    run_gen_model("EleutherAI/gpt-neo-2.7B", empty_df, data, top_obj_to_room_map, top_objs_per_room)
    run_gen_model("xlnet-large-cased", empty_df, data, top_obj_to_room_map, top_objs_per_room)
    run_gen_model("EleutherAI/gpt-j-6B", empty_df, data, top_obj_to_room_map, top_objs_per_room)


if __name__ == "__main__":
    run()

