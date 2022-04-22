import math
import dcor
import sklearn
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn import functional as F
from DataProcessing.DataProcessor import WEATProcessor
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def compute_pred_model_cos(bertprocessor, dataframe, data, top_obj_to_room_map, mode):
    weatprocessor = WEATProcessor()
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    objencodemap = {}
    for obj in all_objs:
        objsent = weatprocessor.get_templates(obj.replace("_", " ").lower())
        objencode = bertprocessor.document_to_vec(objsent, mode, " " + obj.replace("_", " ").lower())
        objencodemap[obj] = objencode

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsent = weatprocessor.get_templates(room.replace("_", " ").lower())
        roomencode = bertprocessor.document_to_vec(roomsent, mode, " " + room.replace("_", " ").lower())

        for obj in all_objs:
            objencode = objencodemap[obj]
            cossim = sklearn.metrics.pairwise.cosine_similarity(roomencode, objencode)
            cossimavg = np.average(cossim)
            cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": cossimavg}, ignore_index=True)
            # cos_df[room][obj] = cossimavg
    return cos_df


def compute_pred_model_distcor(bertprocessor, dataframe, data, top_obj_to_room_map, mode):
    weatprocessor = WEATProcessor()
    # bertprocessor = BertCLSEncoder(model)
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    objencodemap = {}
    for obj in all_objs:
        objsent = weatprocessor.get_templates(obj.replace("_", " ").lower())
        objencode = bertprocessor.document_to_vec(objsent, mode, " " + obj.replace("_", " ").lower())
        objencodemap[obj] = objencode

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsent = weatprocessor.get_templates(room.replace("_", " ").lower())
        roomencode = bertprocessor.document_to_vec(roomsent, mode, " " + room.replace("_", " ").lower())

        for obj in all_objs:
            objencode = objencodemap[obj]
            distcor = dcor.distance_correlation(roomencode, objencode)
            cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": distcor}, ignore_index=True)
            # cos_df[room][obj] = cossimavg
    return cos_df


def compute_pred_model_distcorv2(bertprocessor, dataframe, data, top_obj_to_room_map, mode):
    weatprocessor = WEATProcessor()
    # bertprocessor = BertCLSEncoder(model)
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    objencodemap = {}
    for obj in all_objs:
        objsent = weatprocessor.get_templates(obj.replace("_", " ").lower())
        objencode = bertprocessor.document_to_vec(objsent, mode, " " + obj.replace("_", " ").lower())
        objencodemap[obj] = objencode

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsent = weatprocessor.get_templates(room.replace("_", " ").lower())
        roomencode = bertprocessor.document_to_vec(roomsent, mode, " " + room.replace("_", " ").lower())

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


def compute_pred_model_pred_room_prob(encoder, dataframe, data, top_obj_to_room_map, sentprocessor, pred_rep=0):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    room_encodings = {}
    for room in all_rooms:
        room_enc = encoder.tokenizer.encode(" " + room)
        room_encodings[room] = room_enc[pred_rep]

    log_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])

    for obj in all_objs:
        obj_sent = sentprocessor.fill_template(obj)

        inputs = encoder.tokenizer(obj_sent, return_tensors="pt")
        next_token_logits = encoder.model(**inputs).logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)[0]

        for roomidx, room in enumerate(all_rooms):
            logprob = float(probs[room_encodings[room]])
            log_df = log_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": logprob},
                                   ignore_index=True)
    return log_df


def compute_pred_model_pred_room_log(encoder, dataframe, data, top_obj_to_room_map, sentprocessor, neut_sent, pred_rep=0):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    room_encodings = {}
    for room in all_rooms:
        room_enc = encoder.tokenizer.encode(" " + room)
        room_encodings[room] = room_enc[pred_rep]

    neut_inputs = encoder.tokenizer(neut_sent, return_tensors="pt")
    next_neut_token_logits = encoder.model(**neut_inputs).logits[:, -1, :]
    neut_probs = F.softmax(next_neut_token_logits, dim=-1)[0]

    log_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])

    for obj in all_objs:
        obj_sent = sentprocessor.fill_template(obj)

        inputs = encoder.tokenizer(obj_sent, return_tensors="pt")
        next_token_logits = encoder.model(**inputs).logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)[0]

        for roomidx, room in enumerate(all_rooms):
            logprob = math.log10(probs[room_encodings[room]] / neut_probs[room_encodings[room]])
            log_df = log_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": logprob},
                                   ignore_index=True)
    return log_df


def compute_pred_model_pred_obj_prob(encoder, dataframe, data, top_obj_to_room_map, neut_sent, pred_rep=0):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    obj_encodings = {}
    for obj in all_objs:
        obj_enc = encoder.tokenizer.encode(" " + obj)
        obj_encodings[obj] = obj_enc[pred_rep]

    log_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])

    for room in all_rooms:
        room_sent_1 = neut_sent.format(room=room) + " a"
        room_sent_2 = neut_sent.format(room=room) + " an"

        inputs_1 = encoder.tokenizer(room_sent_1, return_tensors="pt")
        next_token_logits_1 = encoder.model(**inputs_1).logits[:, -1, :]
        probs1 = F.softmax(next_token_logits_1, dim=-1)[0]

        inputs_2 = encoder.tokenizer(room_sent_2, return_tensors="pt")
        next_token_logits_2 = encoder.model(**inputs_2).logits[:, -1, :]
        probs2 = F.softmax(next_token_logits_2, dim=-1)[0]

        for obj_idx, obj in enumerate(all_objs):
            logprob = max(float(probs1[obj_encodings[obj]]), float(probs2[obj_encodings[obj]]))
            log_df = log_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": logprob},
                                   ignore_index=True)
    return log_df


def compute_pred_model_pred_obj_log(encoder, dataframe, data, top_obj_to_room_map, neut_sent, pred_rep=0):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    obj_encodings = {}
    for obj in all_objs:
        obj_enc = encoder.tokenizer.encode(" " + obj)
        obj_encodings[obj] = obj_enc[pred_rep]

    log_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])

    neut_sent_1 = neut_sent.format(room="room") + " a"
    neut_sent_2 = neut_sent.format(room="room") + " an"
    neut_inputs_1 = encoder.tokenizer(neut_sent_1, return_tensors="pt")
    next_neut_token_logits_1 = encoder.model(**neut_inputs_1).logits[:, -1, :]
    neut_probs_1 = F.softmax(next_neut_token_logits_1, dim=-1)[0]
    neut_inputs_2 = encoder.tokenizer(neut_sent_2, return_tensors="pt")
    next_neut_token_logits_2 = encoder.model(**neut_inputs_2).logits[:, -1, :]
    neut_probs_2 = F.softmax(next_neut_token_logits_2, dim=-1)[0]

    for room in all_rooms:
        room_sent_1 = neut_sent.format(room=room) + " a"
        room_sent_2 = neut_sent.format(room=room) + " an"

        inputs_1 = encoder.tokenizer(room_sent_1, return_tensors="pt")
        next_token_logits_1 = encoder.model(**inputs_1).logits[:, -1, :]
        probs1 = F.softmax(next_token_logits_1, dim=-1)[0]

        inputs_2 = encoder.tokenizer(room_sent_2, return_tensors="pt")
        next_token_logits_2 = encoder.model(**inputs_2).logits[:, -1, :]
        probs2 = F.softmax(next_token_logits_2, dim=-1)[0]

        for obj_idx, obj in enumerate(all_objs):
            logprob = max(float(probs1[obj_encodings[obj]]), float(probs2[obj_encodings[obj]]))
            neutprob = max(float(neut_probs_1[obj_encodings[obj]]), float(neut_probs_2[obj_encodings[obj]]))
            logprob = math.log10(logprob / neutprob)
            log_df = log_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": logprob},
                                   ignore_index=True)

    return log_df


def compute_pred_model_classify(bertprocessor, dataframe, data, top_obj_to_room_map, top_objs_per_room, mode):
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
            objencode = bertprocessor.document_to_vec(objsent, mode, " " + obj)
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


def compute_pred_model_classify_all(bertprocessor, dataframe, data, top_obj_to_room_map, top_objs_per_room, mode, model):
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
            objencode = bertprocessor.document_to_vec(objsent, mode, " " + obj)
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
            #clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=100))
            #clf.fit(X_train, y_train)
            #pred = clf.predict(X_test)
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

def compute_pred_model_classify_ffn(bertprocessor, dataframe, data, top_obj_to_room_map, top_objs_per_room, mode, hidden, lr, epochen):
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
            objencode = bertprocessor.document_to_vec(objsent, mode, " " + obj)
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
            # clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=100))
            # clf.fit(X_train, y_train)
            # pred = clf.predict(X_test)
            # with torch.no_grad():
            # print("....")
            pred = net(torch.from_numpy(X_test))
            # print(pred)
            _, pred = torch.max(pred.data, 1)
            # print(pred2)
            pred = pred.detach().numpy()
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