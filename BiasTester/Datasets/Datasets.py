from BiasTester.Datasets.BiasDatasetClass import BiasDataset
import pandas as pd
#pd.set_option('max_columns', None)

class SuncgObjDataset(BiasDataset):
    def __init__(self, path):
        super().__init__(path)
        self.dataset = self.dataset.drop(["None", "Room"])
        self.dataset = self.dataset[self.dataset.Roomcount > 10000]
        self.dataset = self.dataset.iloc[:, 1:].div(self.dataset.Roomcount, axis=0)
        self.dataset = self.get_weighted_objects()


class ScannetObjDataset(BiasDataset):
    def __init__(self, path):
        super().__init__(path)
        self.dataset = self.dataset.drop(["Conference Room"])

        self.dataset = self.dataset[self.dataset.Roomcount > 100]
        self.dataset = self.filter_object_count(2)
        self.dataset = self.dataset.iloc[:, 1:].div(self.dataset.Roomcount, axis=0)

        self.dataset = self.get_weighted_objects()


class NyuV2ObjDataset(BiasDataset):
    def __init__(self, path):
        super().__init__(path)
        self.dataset = self.dataset[self.dataset.Roomcount > 50]
        self.dataset = self.dataset.drop(["dining room"])
        self.dataset = self.dataset.drop(["cable box", "piano", "piano bench", "stairs", "mask"], axis=1)

        self.dataset = self.filter_object_count(10)
        self.dataset = self.dataset.iloc[:, 1:].div(self.dataset.Roomcount, axis=0)
        self.dataset = self.get_weighted_objects()


class PartDataset(BiasDataset):
    def __init__(self, path):
        super().__init__(path)
        self.dataset = self.dataset.drop("muntin", axis=1)
        self.dataset = self.get_weighted_objects()


class WikiHowDataset(BiasDataset):
    def __init__(self, path):
        super().__init__(path, delimiter=",")
        #self.dataset.loc['Total', :] = self.dataset.sum(axis=0)
        self.dataset = self.dataset.fillna(0)
        #self.dataset = self.dataset[self.dataset["sum"] > 20000]
        self.dataset = self.dataset.drop(["mean", "sum"], axis=1)

        self.dataset = self.dataset.transpose()
        self.dataset.loc[:, 'Total'] = self.dataset.sum(axis=1)
        self.dataset = self.dataset[self.dataset.Total > 20]
        self.dataset = self.dataset.drop(["Total"], axis=1)
        #self.dataset = self.dataset.drop(["mean", "Total", "sum", "porn"], axis=1)
        self.dataset = self.filter_object_count(20)

        self.dataset = self.get_weighted_objects()
        self.dataset = self.dataset.transpose()
        #self.dataset = self.dataset.loc[["install for", "read", "watch", "clean up with", "go out for", "listen to", "play", "stay in", "ride"]]
        #self.dataset = self.dataset.loc[:, (self.dataset != 0).any(axis=0)]

if __name__ == "__main__":
    #data = NyuV2ObjDataset("../../1_DataExtraction/nyuv2.csv")
    #data.dataset.to_csv("room-object.csv")

    #data = PartDataset("../../1_DataExtraction/BildWoerterBuch/bildwoerterbuch_selection_v2.csv")
    #data.dataset.to_csv("object-part.csv")

    data = WikiHowDataset(
        "C:/Workspace/Verb2Obj/0_Praktikum/implizite-sprache-resultate-master/corpus_matrix/matrix_data_gold_db/gold/dataframe.csv")
    verbs = ["read", "wash with", "play", "eat", "wear", "listen to"]
    all_objects = []  # [obj]
    for verb in verbs:
        top_objs = data.get_most_important_obj_for_index_n(verb, 10)
        all_objects.extend(top_objs)
    #dataset = data.dataset[all_objects]
    #print(dataset)
    #print("......")
    dataset = data.dataset.loc[verbs, all_objects]
    #print(dataset)
    dataset.to_csv("verb-object.csv")
    exit()
    #dataset = WikiHowDataset(
    #    "F:/Praktikum/5_SS21/Abschlussdokumentationen/Gruppe 1 - Implicite Language/implizite-sprache-resultate-master/implizite-sprache-resultate-master/corpus_matrix/matrix_data_full_db/dataframe.csv"
    #)
    dataset = WikiHowDataset(
        "F:/Praktikum/5_SS21/Abschlussdokumentationen/Gruppe 1 - Implicite Language/implizite-sprache-resultate-master/implizite-sprache-resultate-master/corpus_matrix/matrix_data_gold_db/gold/dataframe.csv"
    )
    #verbs = ["install for", "read", "watch", "clean up with", "go out for", "listen to", "play", "stay in", "ride"]
    verbs = dataset.get_indeces()
    print(dataset.dataset)
    for verb in verbs:
        #print(verb)
        dataset.get_most_important_obj_for_index_n(verb, 5)


    """
    dataset = NyuV2ObjDataset("../../1_DataExtraction/nyuv2.csv")

    dataset.get_most_important_obj_for_index_n("kitchen", 10)
    dataset.get_most_important_obj_for_index_n("office", 10)
    dataset.get_most_important_obj_for_index_n("bathroom", 10)
    dataset.get_most_important_obj_for_index_n("living_room", 10)
    dataset.get_most_important_obj_for_index_n("bedroom", 10)
    dataset.get_most_important_obj_for_index_n("dining_room", 10)
    print("________________")
    """
    #dataset = SuncgObjDataset("../../1_DataExtraction/suncg/results_only_valid_inkl_count_max1.csv")
    #print(dataset.dataset)
    #dataset.get_most_important_obj_for_index_n("Kitchen", 10)
    #print(dataset.dataset["microwave"])
