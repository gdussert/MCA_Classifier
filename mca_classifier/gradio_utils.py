#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import tensor
from torchvision.transforms import InterpolationMode, transforms
import timm
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
timm.layers.set_fused_attn(False)


def create_text_rectangle(text, width=200, height=40, font=cv2.FONT_HERSHEY_SIMPLEX, color_bgr=(102, 163, 255),
                          color_text=(0, 0, 0), text_pad_h=10):
    image = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    font_scale = 1.0
    thickness = 1
    while True:
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        if text_width <= width - 5 and text_height <= height - text_pad_h:
            break
        font_scale -= 0.1
        if font_scale <= 0.1:  # Avoid going to zero
            font_scale = 0.1
            break
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    cv2.putText(image, text, (x, y), font, font_scale, color_text, thickness, cv2.LINE_AA)
    return image


def nms(detections, iou_threshold=0.5):
    if detections is None or len(detections) == 0:
        return detections

    boxes = detections.xyxy
    scores = detections.conf
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return detections[keep]


convert = {
    'acinonyxjubatus': 'Cheetah',
    'aepycerosmelampus': 'Impala',
    'alcelaphusbuselaphus': 'Hartebeest',
    'ardeotiskori': 'Kori Bustard',
    'aves': 'Bird',
    'caracalcaracal': 'Caracal',
    'chlorocebuspygerythrus': 'Vervet Monkey',
    'connochaetestaurinus': 'Wildebeest',
    'crocutacrocuta': 'Spotted Hyena',
    'damaliscuslunatusjimela': 'Topi',
    'equusquagga': 'Zebra',
    'eudorcasthomsonii': "Thomson's Gazelle",
    'felislybica': 'African Wildcat',
    'genetta': 'Genets',
    'giraffacamelopardalis': 'Giraffe',
    'herpestidae': 'Mongooses',
    'hippopotamusamphibius': 'Hippopotamus',
    'hyaenahyaena': 'Striped Hyena',
    'hystrixcristata': 'Porcupine',
    'ictonyxstriatus': 'Polecat',
    'kobusellipsiprymnus': 'Waterbuck',
    'leptailurusserval': 'Serval',
    'lepus': 'Hares',
    'loxodontaafricana': 'Elephant',
    'lupulellamesomelas': 'Jackal',
    'madoqua': 'Dik-diks',
    'mellivoracapensis': 'Honey Badger',
    'nangergranti': "Grant's Gazelle",
    'numididae': 'Guineafowl',
    'orycteropusafer': 'Aardvark',
    'otocyonmegalotis': 'Bat-eared Fox',
    'pantheraleo': 'Lion',
    'pantherapardus': 'Leopard',
    'papio': 'Baboons',
    'phacochoerusafricanus': 'Warthog',
    'protelescristatus': 'Aardwolf',
    'redunca': 'Reedbucks',
    'reptilia': 'Reptiles',
    'rhinocerotidae': 'Rhinoceroses',
    'rodentia': 'Rodents',
    'sagittariusserpentarius': 'Secretarybird',
    'struthiocamelus': 'Ostrich',
    'synceruscaffer': 'Buffalo',
    'tragelaphusoryx': 'Eland',
    'tragelaphusscriptus': 'Bushbuck',
    'viverridae': 'Viverrids'
}
classes = list(convert.values())


def cropSquareCV(imagecv, box):
    x1, y1, x2, y2 = box
    xsize = (x2-x1)
    ysize = (y2-y1)
    if xsize > ysize:
        y1 = y1-int((xsize-ysize)/2)
        y2 = y2+int((xsize-ysize)/2)
    if ysize > xsize:
        x1 = x1-int((ysize-xsize)/2)
        x2 = x2+int((ysize-xsize)/2)
    height, width, _ = imagecv.shape
    croppedimagecv = imagecv[max(0, int(y1)): min(int(y2), height), max(0, int(x1)): min(int(x2), width)]
    return croppedimagecv


classifier_transforms = transforms.Compose([
    transforms.Resize(size=(182, 182), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
    transforms.ToTensor(),
    transforms.Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))])


def get_sns_heatmap(attention_matrix):
    d = attention_matrix.mean(0)
    d_norm = (d - d.min()) / (d.max() - d.min())
    # Plot the heatmap with seaborn
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    sns.heatmap(d_norm, cmap=sns.light_palette("#825f87", as_cmap=True), cbar=False, xticklabels=False, yticklabels=False, ax=ax)
    ax.axis('off')
    # Save the heatmap to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    # Load the image from the buffer
    buf.seek(0)
    heatmap_img = Image.open(buf).convert("RGB")
    heatmap_np = np.array(heatmap_img)
    return heatmap_np


def make_thumbnail_gallery(gallery_images, table_data):
    new_gallery = []
    for i in range(len(gallery_images)):
        im = np.concatenate((np.array(gallery_images[i]), create_text_rectangle(table_data[i][1]),
                             create_text_rectangle(table_data[i][2], color_bgr=(176, 224, 143))))
        im[:30, :30, :] = create_text_rectangle(str(i+1), 30, 30, color_bgr=(0, 0, 0),
                                                color_text=(255, 255, 255), text_pad_h=5)
        new_gallery.append(Image.fromarray(im))
    return new_gallery


def annotate_bbox_on_image(annotated_image, x1, y1, x2, y2, label_text=None):
    height, width, _ = annotated_image.shape
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    if label_text:
        fontScale = fontScale = max(0.5, min(width, height) / 2000)
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
        bg_x1, bg_y1 = x1, max(0, y1 - text_height - baseline - 4)
        bg_x2, bg_y2 = x1 + text_width + 4, y1
        cv2.rectangle(annotated_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.putText(annotated_image, label_text, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fontScale, color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
