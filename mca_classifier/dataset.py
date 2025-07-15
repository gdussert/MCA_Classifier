#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, choice, randint
from mca_classifier.utils import filter_df, class_list


def get_synthetic_seq_list(df, nb_true_seq=2, balance=True, month_and_site=False, seed=None):
    synthetic_seq_list = []
    df["split_key"] = (df.rawfolder.astype(str) + df.hour.astype(str) + df.month.astype(str)) if month_and_site else df.hour
    if seed is not None:
        np.random.seed(seed)
    for sk in df.split_key.unique():
        subdf = df[df.split_key == sk]
        if balance:
            subdf = subdf.groupby('label', group_keys=False)[subdf.columns].apply(lambda x: x.sample(min(len(x), int(subdf.label.value_counts().mean()))))
        seqid = subdf.index.values.astype(int)
        permutations = []
        for i in range(nb_true_seq):
            seqid = np.random.permutation(seqid)
            permutations.append(seqid)
        synthetic_seq_list += [[permutations[i][j] for i in range(nb_true_seq)] for j in range(len(seqid))]
    return synthetic_seq_list


class Dataset(data.Dataset):
    def __init__(self, dataframe, embeddings, classes, min_seqlength=3, fixed_length=True,
                 nb_synthetic_seq=[1, 2, 3], aug=None, balance=True, month_and_site=False, seed=None):
        self.dataframe = dataframe
        self.seed = seed
        self.balance = balance
        self.embeddings = embeddings
        self.labels = {classes[k]: k for k in range(len(classes))}
        self.classes = classes
        self.nb_synthetic_seq = nb_synthetic_seq
        self.min_seqlength = min_seqlength
        self.fixed_length = fixed_length
        self.seq2length = {int(k): v for k, v in self.dataframe["seqlength"].to_dict().items()}
        self.dataframe["hour"] = self.dataframe.date.apply(lambda x: int(x.split(" ")[1][:2]))
        self.dataframe["month"] = self.dataframe.date.apply(lambda x: x.split(" ")[0][:7])
        self.month_and_site = month_and_site
        self.update_synthetic_seqs()
        self.aug = aug if aug else dict(nb=0, mixup=None, gmixup=None, noise=None, gnoise=None, dropout=None, gdropout=None, scale_dropout=None)
        print(self.aug)

    def update_synthetic_seqs(self):
        self.fs = []
        for i in self.nb_synthetic_seq:
            fs = get_synthetic_seq_list(self.dataframe, nb_true_seq=i, balance=self.balance, month_and_site=self.month_and_site, seed=self.seed)
            fs_length = [[self.seq2length[seqid] for seqid in seq] for seq in fs]
            fs = [fs[i] for i in range(len(fs)) if sum(fs_length[i]) >= self.min_seqlength]
            self.fs += fs
        print(f"Dataset has {len(self)} sets, {sum([self.seq2length[x[0]] for x in fs])} crops. ({self.min_seqlength} - {self.nb_synthetic_seq})")

    def __getitem__(self, index):
        imgs, gts, seq_ids, embeddings_ids, aug = [], [], [], [], self.aug
        seqids = self.fs[int(index)]
        list_seq = [self.dataframe.loc[seqid] for seqid in seqids]
        np.random.seed(sum(seqids))
        for i in range(len(seqids)):
            seqlength = self.seq2length[seqids[i]]
            seq_mask = list(range(seqlength))
            if self.fixed_length:
                seq_mask = choice(seq_mask, min(seqlength, self.min_seqlength - len(seqids) + 1), replace=False)
                seq_mask.sort()
            imgs += [list_seq[i]["images"][j] for j in seq_mask]
            gts += [list_seq[i]["gt"][j] for j in seq_mask]
            seq_ids += [seqids[i]]*len(seq_mask)
            embeddings_ids += [list_seq[i]["scores_index"][j] for j in seq_mask]

        if self.fixed_length:
            fs_mask = choice(list(range(len(gts))), self.min_seqlength, replace=False)
            fs_mask.sort()
            imgs = [imgs[i] for i in fs_mask]
            gts = [gts[i] for i in fs_mask]
            seq_ids = [seq_ids[i] for i in fs_mask]
            embeddings_ids = [embeddings_ids[i] for i in fs_mask]

        E = self.embeddings[embeddings_ids]
        seq2count, S = {s: c for s, c in zip(*np.unique(seq_ids, return_counts=True))}, E[0].shape
        idxs = np.random.choice([i for i in range(len(E)) if seq2count[seq_ids[i]] > 1], aug["nb"], replace=False)
        global_idxs = [i for i in range(len(E)) if i not in idxs]
        for i in idxs:
            E[i] = aug["mixup"]*E[i] + (1-aug["mixup"])*self.embeddings[randint(0, self.embeddings.shape[0])] if aug["mixup"] else E[i]
            E[i] = (E[i] * rand(*S) > aug["dropout"]) / ((1 - aug["dropout"]) if aug["scale_dropout"] else 1) if aug["dropout"] else E[i]
            E[i] += np.random.normal(loc=0, scale=aug["noise"], size=S) if aug["noise"] else 0
        for i in global_idxs:
            E[i] = aug["gmixup"]*E[i] + (1-aug["gmixup"])*self.embeddings[randint(0, self.embeddings.shape[0])] if aug["gmixup"] else E[i]
            E[i] = (E[i] * rand(*S) > aug["gdropout"]) / ((1 - aug["gdropout"]) if aug["scale_dropout"] else 1) if aug["gdropout"] else E[i]
            E[i] += np.random.normal(loc=0, scale=aug["gnoise"], size=S) if aug["gnoise"] else 0
        return imgs, E.astype(np.float32), np.array([self.labels[gt] for gt in gts]), np.array(seq_ids)

    def __len__(self):
        return len(self.fs)

    def class_indices(self):
        return self.labels

    def show_sample(self, index, crop_size=(200, 200), path="", img_folder=""):
        imgs, embeddings, labels, seq_ids = self.__getitem__(index)
        show_sample(imgs, np.array(self.classes)[labels], crop_size=crop_size, path=path, seq_ids=seq_ids, img_folder=img_folder)


def show_and_save(img, path="", dpi=192):
    plt.imshow(img)
    plt.axis('off')
    if path:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        plt.show()
    plt.clf()


def show_sample(imgs, gts=None, preds=None, probs=None, crop_size=(200, 200), title="", path="", seq_ids=None, am=None, img_folder=""):
    im_list = []
    for i, img in enumerate(imgs):
        img = img_folder + img
        crop = cv2.resize(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), crop_size, interpolation=cv2.INTER_CUBIC)
        if gts is not None:
            text = str(gts[i])
            if seq_ids is not None:
                text = f"({list(np.unique(seq_ids)).index(seq_ids[i])})" + " " + text
            cv2.putText(crop, text, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, 1)
        if preds is not None:
            if gts is not None:
                color = (255, 0, 0) if gts[i] != preds[i] else (0, 255, 0)
            else:
                color = (0, 0, 255)
            if probs is not None:
                text = str(int(probs[i]*100)) + " " + preds[i]
            else:
                text = preds[i]
            cv2.putText(crop, text, (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, 1)
        im_list.append(crop)
    img = np.concatenate(im_list, axis=1)

    if title:
        cv2.putText(img, title, (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, 1)
    show_and_save(img, path)

    am_img = None
    if am is not None:
        d = am.mean(0)
        d = (np.expand_dims((d - d.min())/(d.max()-d.min()), -1).repeat(3, -1)*255).astype(np.uint8)
        d = cv2.resize(d, crop_size, interpolation=cv2.INTER_NEAREST)
        X, Y, _ = img.shape
        am_img = np.zeros((X+Y, X+Y, 3), dtype=np.uint8)
        am_img[:X, X:X+Y] = img
        am_img[X:X+Y, :X] = np.concatenate(im_list, axis=0)
        am_img[X:, X:] = cv2.resize(d, (Y, Y), interpolation=cv2.INTER_NEAREST)
        show_and_save(am_img, path.split(".")[0] + "_AM." + path.split(".")[1])
    return img, am_img


if __name__ == "__main__":
    import os
    from utils import load_data
    df_imgs, scores, embeddings = load_data(os.path.join(os.path.dirname(__file__), "../data/serengeti_toy"))
    df_imgs, df = filter_df(df_imgs, classes=class_list)

    dataset = Dataset(df, embeddings, class_list, min_seqlength=12, balance=True, nb_synthetic_seq=[2])
    dataset.show_sample(0, img_folder=os.path.join(os.path.dirname(__file__), "../data/crop_images/"))
