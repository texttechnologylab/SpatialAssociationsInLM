import pandas as pd
import seaborn as sns
import dcor
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams['figure.dpi'] = 300


def create_heatmap(df: pd.DataFrame, title: str, subtitle: str = "", colorvalue="dataset_prob", sizevalue="score", cmap="coolwarm", norm = None):
    df = df.sort_values(by=["subset", "object", "room"])

    colorvalue = "score"
    sizevalue = "score"
    # https://github.com/mwaskom/seaborn/issues/2067
    #sns.set(style="dark")
    #correlation = df["dataset_prob"].corr(df["score"])
    #distcor = distance.correlation(df["dataset_prob"], df["score"])
    #distcor = dcor.distance_correlation(df["dataset_prob"], df["score"])

    if norm is None:
        scoremax = df[colorvalue].max()
        scoremin = df[colorvalue].min()
    else:
        scoremax = norm[1]
        scoremin = norm[0]
    # score, dataset_prob
    #plt.figure(1, figsize=(4, 16))
    plt.figure(1, figsize=(4, 10))
    #sns.set(style="dark")
    ax = sns.scatterplot(x="room", y="object",
                         hue=colorvalue, size=sizevalue,
                         hue_norm=(scoremin, scoremax),# size_norm=(0, 1),
                         palette=cmap, sizes=(0, 90),
                         marker="s", linewidth=0, legend=False,
                         data=df)

    norm = plt.Normalize(scoremin, scoremax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.figure.colorbar(sm)
    plt.xticks(rotation=90)
    # plt.title(title)
    plt.title(title + '\n' + subtitle)
    # plt.text(0.5, 1, 'the third line', fontsize=13, ha='center')
    path = GRAPHFOLDERPATH + title + "_" + subtitle + "_heat_colorscore.png"
    path = path.replace(" ", "-")
    plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    GRAPHFOLDERPATH = "results/T/"

    RESULTFILE = ("results/AllNyuResults/bert-large-uncased-obj-cosim.csv", "Cosim Score")
    df = pd.read_csv(RESULTFILE[0], sep='t', index_col=0)
    create_heatmap(df, "BERT-large (Room)", RESULTFILE[1], cmap="rocket_r")

    RESULTFILE = ("results/AllNyuResults/bert-large-uncased-obj-classify-ffn_100_0.01_100.csv", "FFN Classify Score")
    df = pd.read_csv(RESULTFILE[0], sep='t', index_col=0)
    create_heatmap(df, "BERT-large (Room)", RESULTFILE[1], cmap="rocket_r")

    RESULTFILE = ("results/AllNyuResults/bert-large-uncased-mask_room-first_rep-log.csv", "Mask Room Log Score")
    df = pd.read_csv(RESULTFILE[0], sep='t', index_col=0)
    create_heatmap(df, "BERT-large (Room)", RESULTFILE[1], cmap="coolwarm", norm=(-1.5, 1.5))

    RESULTFILE = ("results/AllNyuResults/bert-large-uncased-mask_obj-last_rep-log.csv", "Mask Obj Log Score")
    df = pd.read_csv(RESULTFILE[0], sep='t', index_col=0)
    create_heatmap(df, "BERT-large (Room)", RESULTFILE[1], cmap="coolwarm", norm=(-1.5, 1.5))




