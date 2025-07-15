#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from functools import partial
from timm.models.vision_transformer import Block
import os
import timm
from mca_classifier.utils import make_output_dict
timm.layers.set_fused_attn(False)


class BaseModel(nn.Module):
    def get_head_weights(self):
        params = torch.load(os.path.join(os.path.dirname(__file__), "../models/crop_classifier.pt"), map_location="cpu", weights_only=True)
        params = {k: params[k] for k in ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']}
        self.params = params

    def predict(self, dataloader, device, verbose=True):
        self.eval()
        total_output, gts, imgs, seq_ids, features_list = [], [], [], [], []
        with torch.no_grad():
            for data in tqdm(dataloader, disable=(not verbose)):
                x = data[1].to(device)
                output = self.forward(x).swapaxes(1, 2).softmax(dim=1)
                total_output += output.tolist()
                gts.append(np.array(data[2]))
                imgs.append(np.transpose(data[0]))
                seq_ids.append(np.array(data[3]))
        return make_output_dict(total_output, gts, imgs, seq_ids, features_list, dataloader.batch_size)


class IndepClassifier(BaseModel):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.get_head_weights()
        self.head = nn.Linear(self.params["head.weight"].shape[1], self.params["head.weight"].shape[0])
        self.load_state_dict(self.params, strict=False)

    def forward(self, x):
        x = self.head(x)
        return x


class MCAClassifier(BaseModel):
    def __init__(self, depth=2, num_heads=8):
        super().__init__()
        self.get_head_weights()
        self.embed_dim = self.params["head.weight"].shape[1]
        self.blocks = nn.Sequential(*[Block(dim=self.embed_dim, num_heads=num_heads,
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6))for i in range(depth)])
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.num_classes = self.params["head.weight"].shape[0]
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        self.load_state_dict(self.params, strict=False)

    def forward(self, x):
        x = self.forward_blocks(x)
        x = self.forward_head(x)
        return x

    def forward_blocks(self, x):
        x = self.blocks(x)
        return x

    def forward_head(self, x):
        x = self.head(x)
        return x

    def freeze_head(self):
        self.norm.weight.requires_grad = False
        self.norm.bias.requires_grad = False
        self.head.weight.requires_grad = False
        self.head.bias.requires_grad = False


if __name__ == "__main__":
    from utils import crop_to_seq, load_data, filter_df, class_list
    from dataset import Dataset
    
    device = "cpu"
    model = MCAClassifier(depth=1)
    model.to(device)
    
    df_imgs, scores, embeddings = load_data(os.path.join(os.path.dirname(__file__), "../data/serengeti_toy"))
    df_imgs, df = filter_df(df_imgs, classes=class_list)
    df = crop_to_seq(df_imgs)

    dataset = Dataset(df, embeddings, class_list, min_seqlength=12, fixed_length=True, nb_synthetic_seq=[1, 2, 3])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=1)

    output_dict = model.predict(dataloader, device)
    print(output_dict["acc"])
