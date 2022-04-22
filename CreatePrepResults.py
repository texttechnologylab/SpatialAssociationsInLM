from BiasTester.create_pred_model_results import *
from BiasTester.Encoder.Encoder import BertMASKEncoder

import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams['figure.dpi'] = 300

RESULTFOLDER = "AllPrepResultsV1/"
GRAPHFOLDERPATH = RESULTFOLDER

def create_heatmap(df: pd.DataFrame, title: str, subtitle: str = "", colorvalue="dataset_prob", sizevalue="score", cmap="coolwarm", norm = None):
    df = df.sort_values(by=["relation", "concept", "instance"])
    df["instance_concept"] = df["instance"] + "_" + df["concept"]
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
    plt.figure(1, figsize=(4, 4))
    #sns.set(style="dark")
    ax = sns.scatterplot(x="relation", y="instance_concept",
                         hue=colorvalue, size=sizevalue,
                         hue_norm=(scoremin, scoremax),# size_norm=(0, 1),
                         palette=cmap, sizes=(50,50),
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


def run_mask_model_room_obj(modename, pairs, preps, sent):
    maskencoder = BertMASKEncoder(modename)

    masktoken = maskencoder.tokenizer.mask_token
    masktokenencoded = maskencoder.tokenizer.mask_token_id

    prep_id_list = []
    for prep in preps:
        targetobjenc = maskencoder.tokenizer(" " + prep, add_special_tokens=False)["input_ids"][0]
        prep_id_list.append(targetobjenc)


    log_df = pd.DataFrame(columns=["concept", "instance", "relation", "score"])

    for (inst, conc) in pairs:
        full_mask_sent = sent.format(instance=inst, prep=masktoken, concept=masktoken)
        input = maskencoder.tokenizer.encode_plus(full_mask_sent, return_tensors="pt")
        mask_indeces = []
        for tok_idx, tok_id in enumerate(input["input_ids"][0]):
            if tok_id == masktokenencoded:
                mask_indeces.append(tok_idx)
        mask_indeces = mask_indeces[0]
        output = maskencoder.model(**input)
        logits = output.logits
        full_softmax = F.softmax(logits, dim=-1)
        full_softmax = full_softmax[0, mask_indeces, prep_id_list]


        prep_mask_sent = sent.format(instance=inst, prep=masktoken, concept=conc)
        input = maskencoder.tokenizer.encode_plus(prep_mask_sent, return_tensors="pt")
        mask_indeces = []
        for tok_idx, tok_id in enumerate(input["input_ids"][0]):
            if tok_id == masktokenencoded:
                mask_indeces.append(tok_idx)
        mask_indeces = mask_indeces[0]
        #mask_prep_index = torch.tensor([mask_indeces[0]])
        output = maskencoder.model(**input)
        logits = output.logits
        prep_softmax = F.softmax(logits, dim=-1)
        prep_softmax = prep_softmax[0, mask_indeces, prep_id_list]

        for idx, (p, m) in enumerate(zip(prep_softmax, full_softmax)):
            score = math.log10(p / m)
            log_df = log_df.append({"concept": conc, "instance": inst, "relation": preps[idx], "score": score, "p": p.item()},
                                   ignore_index=True)
    return log_df


def run_mask_model_obj_part(modename, pairs, preps, sent):
    maskencoder = BertMASKEncoder(modename)

    masktoken = maskencoder.tokenizer.mask_token
    masktokenencoded = maskencoder.tokenizer.mask_token_id

    prep_id_list = []
    for prep in preps:
        targetobjenc = maskencoder.tokenizer(" " + prep, add_special_tokens=False)["input_ids"][0]
        prep_id_list.append(targetobjenc)

    log_df = pd.DataFrame(columns=["concept", "instance", "relation", "score"])

    for (inst, conc) in pairs:
        full_mask_sent = sent.format(instance=inst, prep=masktoken + " " + masktoken, concept=masktoken)
        input = maskencoder.tokenizer.encode_plus(full_mask_sent, return_tensors="pt")
        mask_indeces = []
        for tok_idx, tok_id in enumerate(input["input_ids"][0]):
            if tok_id == masktokenencoded:
                mask_indeces.append(tok_idx)
        mask_indeces = mask_indeces[0]
        output = maskencoder.model(**input)
        logits = output.logits
        full_softmax = F.softmax(logits, dim=-1)
        full_softmax = full_softmax[0, mask_indeces, prep_id_list]

        prep_mask_sent = sent.format(instance=inst, prep=masktoken + " " + masktoken, concept=conc)
        input = maskencoder.tokenizer.encode_plus(prep_mask_sent, return_tensors="pt")
        mask_indeces = []
        for tok_idx, tok_id in enumerate(input["input_ids"][0]):
            if tok_id == masktokenencoded:
                mask_indeces.append(tok_idx)
        mask_indeces = mask_indeces[0]

        output = maskencoder.model(**input)
        logits = output.logits
        prep_softmax = F.softmax(logits, dim=-1)
        prep_softmax = prep_softmax[0, mask_indeces, prep_id_list]

        for idx, (p, m) in enumerate(zip(prep_softmax, full_softmax)):
            score = math.log10(p / m)
            log_df = log_df.append({"concept": conc, "instance": inst, "relation": preps[idx], "score": score, "p": p.item()},
                                   ignore_index=True)
    return log_df


def run():
    modelname = 'bert-large-uncased'
    savename = modelname.split("/")[-1]

    pairs = [("toilet", "bathroom"),
             ("toothbrush", "bathroom"),
             ("bed", "bedroom"),
             ("dresser", "bedroom"),
             ("dishwasher", "kitchen"),
             ("stove", "kitchen"),
             ("sofa", "living room"),
             ("television", "living room"),
             ("whiteboard", "office"),
             ("stapler", "office")]
    preps = ["in", "on", "by", "between", "below", "before", "across", "after", "against", "among", "at", "around"]
    sent = "The {instance} is {prep} the {concept}."

    log_obj_df = run_mask_model_room_obj(modelname, pairs, preps, sent)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-prep_obj-room.csv", sep='t')
    create_heatmap(log_obj_df, title="room_object", subtitle= "", norm=(-1.5, 1.5))

    pairs = [("pillow", "bed"),
             ("mattress", "bed"),
             ("pump", "dishwasher"),
             ("motor", "dishwasher"),
             ("lock", "door"),
             ("doorknob", "door"),
             ("keyway", "mortise lock"),
             ("cylinder", "mortise lock"),
             ("egg tray", "refrigerator"),
             ("switch", "refrigerator"),
             ("flush handle", "toilet"),
             ("seat", "toilet")]
    preps = ["part of", "next to", "used by", "close to", "compared to"]
    sent = "The {instance} is {prep} the {concept}."

    log_obj_df = run_mask_model_obj_part(modelname, pairs, preps, sent)
    log_obj_df.to_csv(RESULTFOLDER + "/" + savename + "-prep_part-obj.csv", sep='t')
    create_heatmap(log_obj_df, title="object_part", subtitle="", norm=(-1.5, 1.5))


if __name__ == "__main__":
    run()

