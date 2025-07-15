#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import dateutil
import blosc2
from sklearn.metrics import f1_score, precision_score, recall_score


class_list = ['acinonyxjubatus', 'aepycerosmelampus', 'alcelaphusbuselaphus', 'ardeotiskori', 'aves', 'caracalcaracal',
              'chlorocebuspygerythrus', 'connochaetestaurinus', 'crocutacrocuta', 'damaliscuslunatusjimela', 'equusquagga',
              'eudorcasthomsonii', 'felislybica', 'genetta', 'giraffacamelopardalis', 'herpestidae', 'hippopotamusamphibius',
              'hyaenahyaena', 'hystrixcristata', 'ictonyxstriatus', 'kobusellipsiprymnus', 'leptailurusserval', 'lepus',
              'loxodontaafricana', 'lupulellamesomelas', 'madoqua', 'mellivoracapensis', 'nangergranti', 'numididae',
              'orycteropusafer', 'otocyonmegalotis', 'pantheraleo', 'pantherapardus', 'papio', 'phacochoerusafricanus',
              'protelescristatus', 'redunca', 'reptilia', 'rhinocerotidae', 'rodentia', 'sagittariusserpentarius',
              'struthiocamelus', 'synceruscaffer', 'tragelaphusoryx', 'tragelaphusscriptus', 'viverridae']

safari2024_to_serengeti = {"genettagenetta": "genetta",
                           "connochaetestaurinustaurinus": "connochaetestaurinus",
                           "struthionidae": "struthiocamelus"}


def filter_df(df_imgs, classes=None):
    N = len(df_imgs)
    df_imgs = df_imgs[~df_imgs.date.isna()]
    N2 = len(df_imgs)
    print(N - N2, "crops without date removed")

    if classes is not None:
        df_imgs = df_imgs[df_imgs["gt"].isin(classes)]
    print(N2 - len(df_imgs), "crops with other classes removed")

    df = crop_to_seq(df_imgs)
    return df_imgs, df


def load_data(path):
    df = pd.read_csv(f"{path}/metadata.csv")
    scores = blosc2.load_array(f"{path}/scores.bl2")
    embeddings = blosc2.load_array(f"{path}/embeddings.bl2")
    return df, scores, embeddings


def convert_date(date):
    return dateutil.parser.parse(date)


def crop_to_seq(df):
    df = df[['gt', 'images', 'scores_index', "seqid", "rawfolder", "date"]]
    df_seq = df.groupby("seqid").agg(list)
    df_seq["label"] = df_seq["gt"].apply(lambda lst: max(set(lst), key=lst.count))
    df_seq["seqlength"] = df_seq["gt"].apply(len)
    df_seq["rawfolder"] = df_seq["rawfolder"].apply(lambda lst: max(set(lst), key=lst.count))
    df_seq["date"] = df_seq["date"].apply(lambda x: x[0])
    return df_seq


def compute_ECE(df, n_bins=20, remove_empty=False):
    df = df.sort_values("prob", ignore_index=True)
    bins, acc, nb, ece, bins_ece = [], [], [], 0, []
    for i in range(n_bins):
        idx = df["prob"].between(i/n_bins, (i+1)/n_bins)
        subdf = df[idx]
        if remove_empty and len(subdf) == 0:
            continue
        nb.append(len(subdf))
        bins.append((2*i+1)/(2*n_bins))
        if len(subdf):
            accuracy = (subdf["gt"] == subdf["pred"]).sum()/len(subdf)
            acc.append(accuracy)
            ece += len(subdf)/len(df)*abs(accuracy-subdf["prob"].astype(float).mean())
            bins_ece.append(subdf["prob"].mean())
        else:
            acc, bins_ece = acc + [0], bins_ece + [0]
    return bins, acc, nb, ece, bins_ece


def make_output_dict(total_output, gts, imgs, seq_ids, features_list, batch_size):
    pred = list(map(lambda x: np.array(x).transpose().astype(np.float16), total_output))
    if batch_size == 1:
        imgs = list(map(lambda x: x[0].tolist(), imgs))
        seq_ids = list(map(lambda x: x[0].tolist(), seq_ids))
        gts = list(map(lambda x: np.array(x[0].tolist()), gts))
    else:
        gts = list(np.concatenate(gts))
        seq_ids = list(np.concatenate(seq_ids))
        imgs = list(np.concatenate(imgs))
    output_dict = {"pred": pred, "gt": gts, "img": imgs, "seq_id": seq_ids, "features": features_list}
    p = np.concatenate(pred)
    _, _, _, ece, _ = compute_ECE(pd.DataFrame(dict(pred=p.argmax(1), prob=p.max(1), gt=np.concatenate(gts))))
    output_dict["ece"] = ece
    df = pd.DataFrame({k: output_dict[k] for k in ["gt", "pred", "seq_id"]})
    df["pred_k"] = df.pred.apply(lambda x: x.argmax(1))
    df["pred"] = df.pred.apply(lambda x: x.max(1))
    df = df.explode(["seq_id", "gt", "pred", "pred_k"])
    for k in ["gt", "pred_k"]:
        df[k] = df[k].astype(int)
    output_dict["f1"] = f1_score(df["gt"], df["pred_k"], average="weighted", zero_division=0)
    output_dict["recall"] = recall_score(df["gt"], df["pred_k"], average="weighted", zero_division=0)
    output_dict["precision"] = precision_score(df["gt"], df["pred_k"], average="weighted", zero_division=0)
    df["index"] = df.index
    counts = df.groupby("index")["gt"].agg(list).apply(lambda x: len(np.unique(x))).to_dict()
    df["nbsp"] = df["index"].apply(lambda x: counts[x])
    output_dict["df"] = df
    df1 = df[df.nbsp == 1]
    df2 = df[df.nbsp > 1]
    output_dict["acc_per_nb_sp"] = {1: recall_score(df1["gt"], df1["pred_k"], average="weighted", zero_division=0),
                                    "2+": recall_score(df2["gt"], df2["pred_k"], average="weighted", zero_division=0)}
    output_dict["acc"] = output_dict["recall"]
    output_dict["macro_acc"] = {"all": recall_score(df["gt"], df["pred_k"], average="macro", zero_division=0),
                                1: recall_score(df1["gt"], df1["pred_k"], average="macro", zero_division=0),
                                "2+": recall_score(df2["gt"], df2["pred_k"], average="macro", zero_division=0)}
    output_dict["weighted_f1"] = {"all": f1_score(df["gt"], df["pred_k"], average="weighted", zero_division=0),
                                  1: f1_score(df1["gt"], df1["pred_k"], average="weighted", zero_division=0),
                                  "2+": f1_score(df2["gt"], df2["pred_k"], average="weighted", zero_division=0)}
    return output_dict
