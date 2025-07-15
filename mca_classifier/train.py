#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
from tqdm import tqdm
import torch.nn as nn
from scipy.special import softmax
import torch
import os
import wandb
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from mca_classifier.models import MCAClassifier
from mca_classifier.dataset import Dataset
from mca_classifier.utils import filter_df, class_list, load_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="serengeti_toy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--mixup", type=float, default=0.5)
    parser.add_argument("--gmixup", type=float, default=0.8)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--gnoise", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--gdropout", type=float, default=None)
    parser.add_argument("--scale_dropout", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--nb_aug", type=int, default=4)
    parser.add_argument("--device", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--scheduler_patience", type=int, default=15, help="scheduler patience")
    parser.add_argument("--seqlength", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    wandb_mode = "disabled" if args.disable_wandb else None

    model = MCAClassifier(depth=args.depth)
    model.freeze_head()
    model.to(device)
    optimizer = create_optimizer_v2(model, opt="adamw", lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = Accuracy(task="multiclass", num_classes=model.num_classes)
    acc_fn.to(device)
    model.to(device)
    if args.scheduler_patience:
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=args.scheduler_patience,
                                      threshold=1e-4, cooldown=5)
    df_imgs, scores, embeddings = load_data(os.path.join(os.path.dirname(__file__), f"../data/{args.dataset}"))
    df_imgs, df = filter_df(df_imgs, classes=class_list)

    df["al"] = df.scores_index.apply(lambda x: np.array(class_list)[softmax(scores[x].mean(0)).argmax()])
    df = df[df.al == df.label]

    train_df = df.sample(frac=0.9)
    val_df = df[~df.index.isin(train_df.index)].copy()
    df = train_df.copy()

    aug = dict(nb=args.nb_aug, mixup=args.mixup, gmixup=args.gmixup, noise=args.noise, gnoise=args.gnoise, dropout=args.dropout,
               gdropout=args.gdropout, scale_dropout=args.scale_dropout)
    dataset = Dataset(df, embeddings, class_list, min_seqlength=args.seqlength, aug=aug)

    ds_val = Dataset(val_df, embeddings, class_list, min_seqlength=args.seqlength, aug=aug)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch, num_workers=args.num_workers, shuffle=False)

    config = dict(batch_size=args.batch, epochs=args.epochs, lr=args.lr, opt="adamw", depth=args.depth,
                  scheduler_patience=args.scheduler_patience, aug=aug, seqlength=args.seqlength)
    name = time.strftime("%d-%m-%y_%H-%M-%S", time.localtime(time.time())) if not args.name else args.name

    wandb.init(project="sequence_aware_classifier", config=config, name=name, mode=wandb_mode)
    if not os.path.exists(f"{name}"):
        os.mkdir(f"{name}")

    step, acc_total, loss_total, best_test_acc = 0, 0, 0, 0
    for epoch in range(1, 1 + args.epochs):
        dataset.update_synthetic_seqs()
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, num_workers=args.num_workers, shuffle=True)
        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}/{args.epochs}")
            for data in tepoch:
                x = data[1].to(device)
                y = data[2].to(device)

                optimizer.zero_grad()
                y_hat = model.forward(x).swapaxes(1, 2)
                loss = loss_fn(y_hat, y)
                y_hat_softmax = y_hat.softmax(dim=1)
                train_acc = acc_fn(y_hat_softmax, y)

                loss.backward()
                optimizer.step()

                step += 1
                acc_total += train_acc.item()
                loss_total += loss.item()
                tepoch.set_postfix({"Loss": loss_total / step, "Acc": acc_total / step})

                logs = {"Loss": loss_total / step, "Acc": acc_total / step}
                if (step - 1) % 100 == 0:
                    wandb.log(logs, step=step)

        model.eval()
        val_acc_total, val_loss_total = 0, 0
        for data in tqdm(dl_val, disable=(epoch > 1)):
            with torch.no_grad():
                x = data[1].to(device)
                y = data[2].to(device)

                y_hat = model.forward(x).swapaxes(1, 2)
                val_loss = loss_fn(y_hat, y)
                y_hat_softmax = y_hat.softmax(dim=1)
                val_acc = acc_fn(y_hat_softmax, y)

                val_acc_total += val_acc.item()
                val_loss_total += val_loss.item()

        logs = {"Loss_Val": val_loss_total / len(dl_val), "Acc_Val": val_acc_total / len(dl_val)}
        if args.disable_wandb:
            print(logs)
        if args.scheduler_patience:
            scheduler.step(val_loss_total / len(dl_val))
            logs["lr"] = scheduler._last_lr[0]

        if epoch % 5 == 0 or epoch == 1:
            torch.save(model.state_dict(), f"{name}/ckpt_{epoch}.pt")
        wandb.log(logs, step=step)
        torch.save(model.state_dict(), f"{name}/ckpt.pt")
        if args.scheduler_patience and scheduler._last_lr[0] < 1e-8:
            break
    wandb.finish()
