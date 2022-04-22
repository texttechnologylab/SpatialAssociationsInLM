import pandas as pd
from pandas import DataFrame


class BiasDataset:
    def __init__(self, csvpath, delimiter="\t"):
        self.dataset: DataFrame = pd.read_csv(csvpath, delimiter, header=0, index_col=0)

    def get_objects(self):
        return list(self.dataset.columns)

    def get_indeces(self):
        return list(self.dataset.index)

    def filter_object_count(self, count):
        df = self.dataset.transpose()
        df["objsum"] = df.sum(axis=1)
        df = df[df.objsum > count]
        df = df.transpose()
        df = df.drop("objsum")
        return df

    def get_weighted_objects(self):
        df = self.dataset.transpose()
        df["objsum"] = df.sum(axis=1)
        df = df.iloc[:, :-1].div(df.objsum, axis=0)
        df = df.transpose()
        return df

    def get_most_important_obj_for_index_n(self, index: str, n=5):
        if index in self.get_indeces():
            s = pd.Series(self.dataset.loc[index])
            nlargest = s.nlargest(n)
            # if nlargest[0] > 0.05:
            print(nlargest)
            #    print()
            return list(nlargest.index)
        return None

    def get_least_important_obj_for_index_n(self, index: str, n=5):
        if index in self.get_indeces():
            s = pd.Series(self.dataset.loc[index])
            nlargest = s.nsmallest(n)

            return list(nlargest.index)
        return None

    def save_as_csv(self, savefile):
        self.dataset.to_csv(savefile, sep="\t", decimal=",")

    def __str__(self):
        return str(self.dataset)


if __name__ == "__main__":
    dataset = BiasDataset("../../DataExtraction/results_only_valid_inkl_count_max1.csv")
    print(dataset.get_indeces())
