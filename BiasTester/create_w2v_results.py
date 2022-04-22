import pandas as pd
import numpy as np
import dcor
import torch
import sklearn
import scipy as sp
from scipy.signal.signaltools import correlate
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import mahalanobis
from scipy.spatial import minkowski_distance
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from numpy import dot
from numpy.linalg import norm
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


def compute_w2v_cos(dataframe, data, top_obj_to_room_map, model, rep):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsub = room.replace(" ", "_").lower()
        roomrep = []
        if roomsub in model:
            roomrep.append(model[roomsub])
        else:
            roomsplit = room.lower().split(" ")
            for r in roomsplit:
                if r in model:
                    roomrep.append(model[r])
        for obj in all_objs:
            objsub = obj.replace(" ", "_")
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if len(roomrep) > 0 and len(objrep) > 0:
                if rep == "max":
                    roomnp = np.max(roomrep, 0)
                    objnp = np.max(objrep, 0)
                elif rep == "avg":
                    roomnp = np.average(roomrep, 0)
                    objnp = np.average(objrep, 0)
                else:
                    roomnp = np.min(roomrep, 0)
                    objnp = np.min(objrep, 0)

                cosim = dot(roomnp, objnp) / (norm(roomnp) * norm(objnp))

                cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": cosim}, ignore_index=True)
            else:
                print("!!!!!!!!!!")
                print(room)
                print(obj)
    return cos_df


def compute_w2v_distcor(dataframe, data, top_obj_to_room_map, model, rep):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsub = room.replace(" ", "_").lower()
        roomrep = []
        if roomsub in model:
            roomrep.append(model[roomsub])
        else:
            roomsplit = room.lower().split(" ")
            for r in roomsplit:
                if r in model:
                    roomrep.append(model[r])
        for obj in all_objs:
            objsub = obj.replace(" ", "_")
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if len(roomrep) > 0 and len(objrep) > 0:
                if rep == "max":
                    roomnp = np.max(roomrep, 0)
                    objnp = np.max(objrep, 0)
                elif rep == "avg":
                    roomnp = np.average(roomrep, 0)
                    objnp = np.average(objrep, 0)
                else:
                    roomnp = np.min(roomrep, 0)
                    objnp = np.min(objrep, 0)

                cosim = dcor.distance_correlation(roomnp, objnp)

                cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": cosim}, ignore_index=True)
            else:
                print("!!!!!!!!!!")
                print(room)
                print(obj)
    return cos_df


def compute_mahalanobis(array_1, array_2):
    #https://www.machinelearningplus.com/statistics/mahalanobis-distance/
    #https: // stackoverflow.com / questions / 53296903 / how - to - find - mahalanobis - distance - between - two - 1d - arrays - in -python
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    V = np.cov(np.array([array_1, array_2]).T)
    IV = np.linalg.pinv(V)
    return mahalanobis(array_1, array_2, IV)


def compute_w2v_mahalanobis(dataframe, data, top_obj_to_room_map, model, rep):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsub = room.replace(" ", "_").lower()
        roomrep = []
        if roomsub in model:
            roomrep.append(model[roomsub])
        else:
            roomsplit = room.lower().split(" ")
            for r in roomsplit:
                if r in model:
                    roomrep.append(model[r])
        for obj in all_objs:
            objsub = obj.replace(" ", "_")
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if len(roomrep) > 0 and len(objrep) > 0:
                if rep == "max":
                    roomnp = np.max(roomrep, 0)
                    objnp = np.max(objrep, 0)
                elif rep == "avg":
                    roomnp = np.average(roomrep, 0)
                    objnp = np.average(objrep, 0)
                else:
                    roomnp = np.min(roomrep, 0)
                    objnp = np.min(objrep, 0)


                cosim = compute_mahalanobis(roomnp, objnp)

                cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": cosim}, ignore_index=True)
            else:
                print("!!!!!!!!!!")
                print(room)
                print(obj)
    return cos_df


def compute_w2v_minkowski(dataframe, data, top_obj_to_room_map, model, rep):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsub = room.replace(" ", "_").lower()
        roomrep = []
        if roomsub in model:
            roomrep.append(model[roomsub])
        else:
            roomsplit = room.lower().split(" ")
            for r in roomsplit:
                if r in model:
                    roomrep.append(model[r])
        for obj in all_objs:
            objsub = obj.replace(" ", "_")
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if len(roomrep) > 0 and len(objrep) > 0:
                if rep == "max":
                    roomnp = np.max(roomrep, 0)
                    objnp = np.max(objrep, 0)
                elif rep == "avg":
                    roomnp = np.average(roomrep, 0)
                    objnp = np.average(objrep, 0)
                else:
                    roomnp = np.min(roomrep, 0)
                    objnp = np.min(objrep, 0)


                cosim = minkowski_distance(roomnp, objnp)

                cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": cosim}, ignore_index=True)
            else:
                print("!!!!!!!!!!")
                print(room)
                print(obj)
    return cos_df



def compute_w2v_pearson(dataframe, data, top_obj_to_room_map, model, rep):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsub = room.replace(" ", "_").lower()
        roomrep = []
        if roomsub in model:
            roomrep.append(model[roomsub])
        else:
            roomsplit = room.lower().split(" ")
            for r in roomsplit:
                if r in model:
                    roomrep.append(model[r])
        for obj in all_objs:
            objsub = obj.replace(" ", "_")
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if len(roomrep) > 0 and len(objrep) > 0:
                if rep == "max":
                    roomnp = np.max(roomrep, 0)
                    objnp = np.max(objrep, 0)
                elif rep == "avg":
                    roomnp = np.average(roomrep, 0)
                    objnp = np.average(objrep, 0)
                else:
                    roomnp = np.min(roomrep, 0)
                    objnp = np.min(objrep, 0)

                #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
                cosim = pearsonr(roomnp, objnp)[0]

                cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": cosim}, ignore_index=True)
            else:
                print("!!!!!!!!!!")
                print(room)
                print(obj)
    return cos_df


def compute_w2v_spearman(dataframe, data, top_obj_to_room_map, model, rep):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsub = room.replace(" ", "_").lower()
        roomrep = []
        if roomsub in model:
            roomrep.append(model[roomsub])
        else:
            roomsplit = room.lower().split(" ")
            for r in roomsplit:
                if r in model:
                    roomrep.append(model[r])
        for obj in all_objs:
            objsub = obj.replace(" ", "_")
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if len(roomrep) > 0 and len(objrep) > 0:
                if rep == "max":
                    roomnp = np.max(roomrep, 0)
                    objnp = np.max(objrep, 0)
                elif rep == "avg":
                    roomnp = np.average(roomrep, 0)
                    objnp = np.average(objrep, 0)
                else:
                    roomnp = np.min(roomrep, 0)
                    objnp = np.min(objrep, 0)

                #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
                cosim = spearmanr(roomnp, objnp)[0]

                cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": cosim}, ignore_index=True)
            else:
                print("!!!!!!!!!!")
                print(room)
                print(obj)
    return cos_df

def compute_w2v_kendalltau(dataframe, data, top_obj_to_room_map, model, rep):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

    cos_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for room in all_rooms:
        roomsub = room.replace(" ", "_").lower()
        roomrep = []
        if roomsub in model:
            roomrep.append(model[roomsub])
        else:
            roomsplit = room.lower().split(" ")
            for r in roomsplit:
                if r in model:
                    roomrep.append(model[r])
        for obj in all_objs:
            objsub = obj.replace(" ", "_")
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if len(roomrep) > 0 and len(objrep) > 0:
                if rep == "max":
                    roomnp = np.max(roomrep, 0)
                    objnp = np.max(objrep, 0)
                elif rep == "avg":
                    roomnp = np.average(roomrep, 0)
                    objnp = np.average(objrep, 0)
                else:
                    roomnp = np.min(roomrep, 0)
                    objnp = np.min(objrep, 0)

                #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
                cosim = kendalltau(roomnp, objnp)[0]

                cos_df = cos_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                    "dataset_prob": data.dataset[obj][room], "score": cosim}, ignore_index=True)
            else:
                print("!!!!!!!!!!")
                print(room)
                print(obj)
    return cos_df



def compute_w2v_classify(dataframe, data, top_obj_to_room_map, top_objs_per_room, model, rep):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

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
            objsub = obj.replace(" ", "_").lower()
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.lower().split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if rep == "max":
                X.append(np.max(objrep, 0))
            elif rep == "avg":
                X.append(np.average(objrep, 0))
            else:
                X.append(np.min(objrep, 0))

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
                                              "dataset_prob": data.dataset[obj][room], "score": float(val_size)},
                                             ignore_index=True)
    return classify_df


def compute_w2v_classify_all(dataframe, data, top_obj_to_room_map, top_objs_per_room, model, rep, classifier):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

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
            objsub = obj.replace(" ", "_").lower()
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.lower().split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if rep == "max":
                X.append(np.max(objrep, 0))
            elif rep == "avg":
                X.append(np.average(objrep, 0))
            elif rep == "min":
                X.append(np.min(objrep, 0))
            else:
                print("ERROR!!!")
                exit()

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

            clf = sklearn.base.clone(classifier)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)

            pred_room = idx_to_room_map[pred[0]]
            epochcacc.append(int(pred[0] == y_test[0]))

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
                                              "dataset_prob": data.dataset[obj][room], "score": float(val_size)},
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


def compute_w2v_classify_ffn(dataframe, data, top_obj_to_room_map, top_objs_per_room, model, rep, hidden, lr, epochen):
    all_rooms = dataframe.columns.tolist()
    all_objs = dataframe.index.tolist()

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
            objsub = obj.replace(" ", "_").lower()
            objrep = []
            if objsub in model:
                objrep.append(model[objsub])
            else:
                objsplit = obj.lower().split(" ")
                for r in objsplit:
                    if r in model:
                        objrep.append(model[r])

            if rep == "max":
                X.append(np.max(objrep, 0))
            elif rep == "avg":
                X.append(np.average(objrep, 0))
            elif rep == "min":
                X.append(np.min(objrep, 0))
            else:
                print("ERROR!!!")
                exit()

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
        acclist.append(sum(epochcacc) / float(len(epochcacc)))

    classify_df = pd.DataFrame(columns=["room", "object", "subset", "dataset_prob", "score"])
    for obj in obj_room_pred.keys():
        for room in all_rooms:
            val_size = 0
            if room in obj_room_pred[obj].keys():
                val_size = obj_room_pred[obj][room]

            classify_df = classify_df.append({"room": room, "object": obj, "subset": top_obj_to_room_map[obj],
                                              "dataset_prob": data.dataset[obj][room], "score": float(val_size)},
                                             ignore_index=True)
    print(acclist)
    print(float(sum(acclist)) / float(len(acclist)))
    return classify_df
