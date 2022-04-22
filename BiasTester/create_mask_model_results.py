import sklearn
import torch
import math
import pandas as pd
import numpy as np
import dcor
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from DataProcessing.DataProcessor import WEATProcessor


def compute_mask_model_cos(bertprocessor, dataframe, data, top_obj_to_room_map, mode):
    weatprocessor = WEATProcessor()
    # bertprocessor = BertCLSEncoder(model)
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    objencodemap = {}
    for obj in all_objs:
        objsent = weatprocessor.get_templates(obj.replace("_", " ").lower())
        objencode = bertprocessor.documents_to_vecs(objsent, mode, " " + obj.replace("_", " ").lower())
        objencodemap[obj] = objencode

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsent = weatprocessor.get_templates(room.replace("_", " ").lower())
        roomencode = bertprocessor.documents_to_vecs(roomsent, mode, " " + room.replace("_", " ").lower())

        for obj in all_objs:
            objencode = objencodemap[obj]
            cossim = sklearn.metrics.pairwise.cosine_similarity(roomencode, objencode)
            cossimavg = np.average(cossim)
            cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": cossimavg}, ignore_index=True)
            # cos_df[room][obj] = cossimavg
    return cos_df


def compute_mask_model_distcor(bertprocessor, dataframe, data, top_obj_to_room_map, mode):
    weatprocessor = WEATProcessor()
    # bertprocessor = BertCLSEncoder(model)
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    objencodemap = {}
    for obj in all_objs:
        objsent = weatprocessor.get_templates(obj.replace("_", " ").lower())
        objencode = bertprocessor.documents_to_vecs(objsent, mode, " " + obj.replace("_", " ").lower())
        objencodemap[obj] = objencode

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsent = weatprocessor.get_templates(room.replace("_", " ").lower())
        roomencode = bertprocessor.documents_to_vecs(roomsent, mode, " " + room.replace("_", " ").lower())

        for obj in all_objs:
            objencode = objencodemap[obj]
            distcor = dcor.distance_correlation(roomencode, objencode)
            cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": distcor}, ignore_index=True)
            # cos_df[room][obj] = cossimavg
    return cos_df


def compute_mask_model_distcorv2(bertprocessor, dataframe, data, top_obj_to_room_map, mode):
    weatprocessor = WEATProcessor()
    # bertprocessor = BertCLSEncoder(model)
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    objencodemap = {}
    for obj in all_objs:
        objsent = weatprocessor.get_templates(obj.replace("_", " ").lower())
        objencode = bertprocessor.documents_to_vecs(objsent, mode, " " + obj.replace("_", " ").lower())
        objencodemap[obj] = objencode

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsent = weatprocessor.get_templates(room.replace("_", " ").lower())
        roomencode = bertprocessor.documents_to_vecs(roomsent, mode, " " + room.replace("_", " ").lower())

        for obj in all_objs:
            all_dcor = []
            objencode = objencodemap[obj]
            for roomvec in roomencode.numpy():
                for objvec in objencode.numpy():
                    distcor = dcor.distance_correlation(np.array([roomvec]), np.array([objvec]))
                    all_dcor.append(distcor)
            all_dcor_avg = np.average(all_dcor)
            cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": all_dcor_avg}, ignore_index=True)
            # cos_df[room][obj] = cossimavg
    return cos_df


def compute_mask_model_mask_instance_log(encoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1):
    all_concepts = dataframe.columns.tolist()
    all_instances = dataframe.index.tolist()

    #encoder = BertMASKEncoder(model)
    #encoder.model = RobertaForMaskedLM.from_pretrained("roberta-large", return_dict=True)
    masktoken = encoder.tokenizer.mask_token
    masktokenencoded = encoder.tokenizer.mask_token_id

    allmask = sentprocessor.fill_template(masktoken, masktoken)

    maskworddict = {}
    for room in all_concepts:
        objmmask = sentprocessor.fill_template(masktoken, room.replace("_", " ").lower())
        #print(objmmask)
        input = encoder.tokenizer.encode_plus(objmmask, return_tensors="pt")
        output = encoder.model(**input)
        logits = output.logits
        softmax = F.softmax(logits, dim=-1)

        mask_indeces = []
        for tok_idx, tok_id in enumerate(input["input_ids"][0]):
            if tok_id == masktokenencoded:
                mask_indeces.append(tok_idx)
        mask_index = torch.tensor([mask_indeces[0]])
        maskworddict[room] = softmax[0, mask_index, :]
        # mask_word =

    input = encoder.tokenizer.encode_plus(allmask, return_tensors="pt")
    output = encoder.model(**input)
    logits = output.logits
    softmax = F.softmax(logits, dim=-1)

    mask_indeces = []
    for tok_idx, tok_id in enumerate(input["input_ids"][0]):
        if tok_id == masktokenencoded:
            mask_indeces.append(tok_idx)
    mask_all_index = torch.tensor([mask_indeces[0]])
    all_mask_word = softmax[0, mask_all_index, :]

    log_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])

    for room in all_concepts:
        for targetobj in all_instances:
            targetobj = targetobj.replace("_", " ").lower()
            targetobjenc = encoder.tokenizer(" " + targetobj, add_special_tokens=False)["input_ids"][rep]
            targetobjprob = maskworddict[room][0][targetobjenc]
            targetobjproball = all_mask_word[0][targetobjenc]

            logprob = math.log10(targetobjprob / targetobjproball)
            log_df = log_df.append({"room": room, "object": targetobj, "subset": top_obj_to_room_map[targetobj],
                                    "dataset_prob": data.dataset[targetobj][room], "score": logprob},
                                   ignore_index=True)
    return log_df


def compute_mask_model_mask_concept_log(encoder, dataframe, data, top_obj_to_room_map, sentprocessor, rep=-1):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    # encoder = BertMASKEncoder(model)
    #encoder.model = RobertaForMaskedLM.from_pretrained("roberta-large", return_dict=True)
    masktoken = encoder.tokenizer.mask_token
    masktokenencoded = encoder.tokenizer.mask_token_id

    allmask = sentprocessor.fill_template(masktoken, masktoken)

    input = encoder.tokenizer.encode_plus(allmask, return_tensors="pt")
    output = encoder.model(**input)
    logits = output.logits
    softmax = F.softmax(logits, dim=-1)

    mask_indeces = []
    for tok_idx, tok_id in enumerate(input["input_ids"][0]):
        if tok_id == masktokenencoded:
            mask_indeces.append(tok_idx)
    mask_all_index = torch.tensor([mask_indeces[-1]])
    all_mask_word = softmax[0, mask_all_index, :]

    log_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        targetroomenc = encoder.tokenizer(" " + room.replace("_", " ").lower(), add_special_tokens=False)["input_ids"][rep]

        for targetobj in all_objs:
            targetobj = targetobj.replace("_", " ").lower()

            roommask = sentprocessor.fill_template(targetobj, masktoken)
            #print(roommask)
            input = encoder.tokenizer.encode_plus(roommask, return_tensors="pt")
            output = encoder.model(**input)
            logits = output.logits
            softmax = F.softmax(logits, dim=-1)

            mask_indeces = []
            for tok_idx, tok_id in enumerate(input["input_ids"][0]):
                if tok_id == masktokenencoded:
                    mask_indeces.append(tok_idx)
            room_mask_index = torch.tensor([mask_indeces[-1]])
            room_mask_word = softmax[0, room_mask_index, :]

            targetobjprob = room_mask_word[0][targetroomenc]
            targetobjproball = all_mask_word[0][targetroomenc]

            logprob = math.log10(targetobjprob / targetobjproball)
            log_df = log_df.append({"room": room, "object": targetobj, "subset": top_obj_to_room_map[targetobj],
                                    "dataset_prob": data.dataset[targetobj][room], "score": logprob},
                                   ignore_index=True)
    return log_df


def compute_mask_model_classify(bertprocessor, dataframe, data, top_obj_to_room_map, top_objs_per_room, mode):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    weatprocessor = WEATProcessor()
    #bertprocessor = BertCLSEncoder(model)

    X = []
    Y_Gold = []
    objlist = []

    obj_goldroom_map = {}
    obj_room_pred = {}
    idx_to_room_map = {}
    for roomidx, room in enumerate(top_objs_per_room.keys()):
        topobjs = top_objs_per_room[room]
        idx_to_room_map[roomidx] = room
        for obj in topobjs:
            objsent = weatprocessor.get_templates(obj.replace("_", " ").lower())
            objencode = bertprocessor.documents_to_vecs(objsent, mode, " " + obj)
            X.append(np.average(objencode, 0))
            Y_Gold.append(roomidx)
            objlist.append(obj)
            # roomlist.append(room)
            obj_room_pred[obj] = {}
            obj_goldroom_map[obj] = room

    X = np.array(X)
    Y_Gold = np.array(Y_Gold)
    objlist = np.array(objlist)
    # roomlist = np.array(roomlist)
    acclist = []
    for iter in range(100):
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        epochcacc = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y_Gold[train_index], Y_Gold[test_index]
            _, obj_test = objlist[train_index], objlist[test_index]
            # _, room_test = roomlist[train_index], roomlist[test_index]
            clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=100))
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            pred_room = idx_to_room_map[pred[0]]
            epochcacc.append(int(pred[0] == y_test[0]))
            # print(pred)
            # print(y_test)
            if pred_room in obj_room_pred[obj_test[0]]:
                obj_room_pred[obj_test[0]][pred_room] += 1
            else:
                obj_room_pred[obj_test[0]][pred_room] = 0
        acclist.append(sum(epochcacc) / float(len(epochcacc)))

    classify_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for obj in obj_room_pred.keys():
        for room in all_rooms:
            val_size = 0
            if room in obj_room_pred[obj].keys():
                val_size = obj_room_pred[obj][room]

            classify_df = classify_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                              "dataset_prob": data.dataset[obj][room], "score": val_size},
                                             ignore_index=True)
    return classify_df


def compute_mask_model_classify_all(bertprocessor, dataframe, data, top_obj_to_room_map, top_objs_per_room, mode, model):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    weatprocessor = WEATProcessor()
    #bertprocessor = BertCLSEncoder(model)

    X = []
    Y_Gold = []
    objlist = []

    obj_goldroom_map = {}
    obj_room_pred = {}
    idx_to_room_map = {}
    for roomidx, room in enumerate(top_objs_per_room.keys()):
        topobjs = top_objs_per_room[room]
        idx_to_room_map[roomidx] = room
        for obj in topobjs:
            objsent = weatprocessor.get_templates(obj.replace("_", " ").lower())
            objencode = bertprocessor.documents_to_vecs(objsent, mode, " " + obj)
            X.append(np.average(objencode, 0))
            Y_Gold.append(roomidx)
            objlist.append(obj)
            # roomlist.append(room)
            obj_room_pred[obj] = {}
            obj_goldroom_map[obj] = room

    X = np.array(X)
    #X = StandardScaler().fit_transform(X)
    Y_Gold = np.array(Y_Gold)
    objlist = np.array(objlist)
    # roomlist = np.array(roomlist)
    acclist = []
    for iter in range(100):
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        epochcacc = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y_Gold[train_index], Y_Gold[test_index]
            _, obj_test = objlist[train_index], objlist[test_index]
            # _, room_test = roomlist[train_index], roomlist[test_index]
            clf = sklearn.base.clone(model)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            pred_room = idx_to_room_map[pred[0]]
            epochcacc.append(int(pred[0] == y_test[0]))
            # print(pred)
            # print(y_test)
            if pred_room in obj_room_pred[obj_test[0]]:
                obj_room_pred[obj_test[0]][pred_room] += 1
            else:
                obj_room_pred[obj_test[0]][pred_room] = 0
        acclist.append(sum(epochcacc) / float(len(epochcacc)))

    classify_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for obj in obj_room_pred.keys():
        for room in all_rooms:
            val_size = 0
            if room in obj_room_pred[obj].keys():
                val_size = obj_room_pred[obj][room]

            classify_df = classify_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                              "dataset_prob": data.dataset[obj][room], "score": val_size},
                                             ignore_index=True)
    print(acclist)
    print(float(sum(acclist)) / float(len(acclist)))

    return classify_df


class Net(nn.Module):
    def __init__(self, dim, classes, hidden):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def compute_mask_model_classify_ffn(bertprocessor, dataframe, data, top_obj_to_room_map, top_objs_per_room, mode, hidden=10, epochen=10, lr=0.001):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    weatprocessor = WEATProcessor()
    #bertprocessor = BertCLSEncoder(model)

    X = []
    Y_Gold = []
    objlist = []

    obj_goldroom_map = {}
    obj_room_pred = {}
    idx_to_room_map = {}
    for roomidx, room in enumerate(top_objs_per_room.keys()):
        topobjs = top_objs_per_room[room]
        idx_to_room_map[roomidx] = room
        for obj in topobjs:
            objsent = weatprocessor.get_templates(obj.replace("_", " ").lower())
            objencode = bertprocessor.documents_to_vecs(objsent, mode, " " + obj)
            X.append(np.average(objencode, 0))
            Y_Gold.append(roomidx)
            #ourarray = np.zeros(len(all_rooms))
            #ourarray[roomidx] = 1
            #Y_Gold.append(ourarray)
            objlist.append(obj)
            # roomlist.append(room)
            obj_room_pred[obj] = {}
            obj_goldroom_map[obj] = room

    X = np.array(X)
    Y_Gold = np.array(Y_Gold)
    objlist = np.array(objlist)
    # roomlist = np.array(roomlist)
    acclist = []
    for iter in range(100):
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        epochcacc = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y_Gold[train_index], Y_Gold[test_index]
            _, obj_test = objlist[train_index], objlist[test_index]

            net = Net(X_train.shape[1], len(all_rooms), hidden)
            net.train()
            optimizer = optim.Adam(net.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochen):
                net.zero_grad()
                output = net(torch.from_numpy(X_train))

                loss = criterion(output, torch.from_numpy(y_train).long())
                loss.backward()
                optimizer.step()

            # _, room_test = roomlist[train_index], roomlist[test_index]
            #clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=100))
            #clf.fit(X_train, y_train)
            #pred = clf.predict(X_test)
            #with torch.no_grad():
                #print("....")
            pred = net(torch.from_numpy(X_test))
                #print(pred)
            _, pred = torch.max(pred.data, 1)
               #print(pred2)
            pred = pred.detach().numpy()
            pred_room = idx_to_room_map[pred[0]]
            epochcacc.append(int(pred[0] == y_test[0]))
            # print(pred)
            # print(y_test)
            if pred_room in obj_room_pred[obj_test[0]]:
                obj_room_pred[obj_test[0]][pred_room] += 1
            else:
                obj_room_pred[obj_test[0]][pred_room] = 0
        #print(epochcacc)
        #print(len(epochcacc))
        acclist.append(sum(epochcacc) / float(len(epochcacc)))

    classify_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for obj in obj_room_pred.keys():
        for room in all_rooms:
            val_size = 0
            if room in obj_room_pred[obj].keys():
                val_size = obj_room_pred[obj][room]

            classify_df = classify_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                              "dataset_prob": data.dataset[obj][room], "score": val_size},
                                                ignore_index=True)
    print(acclist)
    print(float(sum(acclist)) / float(len(acclist)))
    return classify_df