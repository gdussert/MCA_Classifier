#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
from mca_classifier.models import IndepClassifier, MCAClassifier
from mca_classifier.dataset import Dataset
from mca_classifier.utils import filter_df, safari2024_to_serengeti, class_list, load_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="serengeti_toy")
    parser.add_argument("--model", type=str, default="../models/mca_classifier.pt")
    parser.add_argument("--dataset_type", type=str, default="real", choices=["real", "synthetic"])
    parser.add_argument("--seed", type=int, default=0, help="seed for synthetic dataset type")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    model_path = args.model
    device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    name = args.model.split("/")[-1].replace(".pt", "")

    model = MCAClassifier(depth=args.depth)
    model.freeze_head()
    model.to(device)
    model_ic = IndepClassifier()
    model_ic.to(device)

    df_imgs, scores, embeddings = load_data(os.path.join(os.path.dirname(__file__), f"../data/{args.dataset}"))
    if args.dataset == "safari2024":
        df_imgs["gt"] = df_imgs["gt"].apply(lambda x: safari2024_to_serengeti.setdefault(x, x))
    df_imgs, df = filter_df(df_imgs, classes=class_list)

    if args.dataset_type == "real":
        ds_test = Dataset(df[np.logical_and(df.seqlength >= 1, df.seqlength <= 1024)].copy(), embeddings, class_list, min_seqlength=1,
                          fixed_length=False, nb_synthetic_seq=[1], balance=False)
    elif args.dataset_type == "synthetic":
        ds_test = Dataset(df[np.logical_and(df.seqlength >= 1, df.seqlength <= 1024)].copy(), embeddings, class_list, min_seqlength=1,
                          fixed_length=False, nb_synthetic_seq=[2], balance=False, month_and_site=True, seed=args.seed)
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=1, num_workers=args.num_workers, shuffle=False)

    model.eval()
    model.load_state_dict(torch.load(args.model, weights_only=True, map_location=device))

    def format_res(res):
        acc = np.round(res["acc"]*100, 3)
        ece = np.round(res["ece"]*100, 3)

        def rd(x):
            return str({k: round(x[k], 3) for k in x})
        return str(acc) + "% " + str(ece) + "% " + rd(res["acc_per_nb_sp"])

    res_sa = model.predict(test_dataloader, device)
    print("SA", format_res(res_sa))
    res_ic = model_ic.predict(test_dataloader, device)
    print("IC", format_res(res_ic))
