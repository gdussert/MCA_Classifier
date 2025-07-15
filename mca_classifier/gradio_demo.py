import gradio as gr
import cv2
from PIL import Image
import timm
from ultralytics import YOLO
import numpy as np
import torch
from PIL import Image as PILImage
from timm.utils.attention_extract import AttentionExtract
from glob import glob
import os
import argparse
from mca_classifier.gradio_utils import classes, cropSquareCV, nms, classifier_transforms, get_sns_heatmap
from mca_classifier.gradio_utils import make_thumbnail_gallery, annotate_bbox_on_image
from mca_classifier.models import MCAClassifier


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str)
args = parser.parse_args()

device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
dirname = os.path.dirname(__file__)

detector = YOLO(os.path.join(dirname, '../models/MDV6-yolov10-e-1280.pt'))

classifier = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=False, num_classes=46, dynamic_img_size=True)
classifier.load_state_dict(torch.load(os.path.join(dirname, '../models/crop_classifier.pt'), map_location=device, weights_only=True))
classifier.to(device)

seq_model = MCAClassifier(depth=4)
seq_model.load_state_dict(torch.load(os.path.join(dirname, '../models/mca_classifier.pt'), map_location=device, weights_only=True))
seq_model.to(device)
extractor = AttentionExtract(seq_model, method='fx')  # for the attention matrix


def show_preds_image(image1, image2=None):
    image_paths = [image1, image2]
    predictions = []
    embeddings_seq = []

    gallery_images = []
    table_data = []
    img_list = []
    crop_size = (200, 200)

    # detection and independent classifier + save embeddings for mca classsifer
    for image_path in image_paths:
        if image_path is None:
            continue
        image = cv2.imread(image_path)
        outputs = detector.predict(source=image_path, imgsz=1280, verbose=True, conf=0.25, device=device)
        detections = nms(outputs[0].cpu().numpy().boxes, 0.5)

        for i, (box_cls, conf, box) in enumerate(zip(detections.cls, detections.conf, detections.xyxy)):
            if conf < 0.25:
                continue

            x1, y1, x2, y2 = map(int, box)
            croppedimagecv = cropSquareCV(image, box.copy())[:, :, (2, 1, 0)]
            croppedimage = Image.fromarray(croppedimagecv)  # for classifier
            im = cv2.resize(croppedimagecv, crop_size, interpolation=cv2.INTER_CUBIC)  # for visualization
            gallery_images.append(PILImage.fromarray(im))  # for gallery
            img_list.append(im)  # for attention matrix
            if box_cls == 0:
                x = classifier_transforms(croppedimage).unsqueeze(dim=0).to(device)
                with torch.no_grad():
                    embeddings = classifier.forward_features(x)
                    embeddings_seq.append(embeddings[:, 0:1, :])
                    pred = classifier.forward_head(embeddings).softmax(dim=1).cpu()

                scores = {classes[j]: float(pred[0][j]) for j in range(len(classes))}
                top_class = max(scores, key=scores.get)
                label_text = f"{top_class}: {scores[top_class]*100:.0f}%"
            else:
                if box_cls == 1:
                    label_text = "human"
                elif box_cls == 2:
                    label_text = "vehicle"

            predictions.append([(x1, y1, x2, y2), label_text, image_path])

    # sequence-aware classifier
    if len(embeddings_seq) > 0:
        x = torch.concat(embeddings_seq, 1)
        with torch.no_grad():
            embeddings = seq_model.forward_blocks(x)
            pred = classifier.forward_head(embeddings.swapaxes(0, 1)).cpu().softmax(dim=1)
            oo = extractor(x)
            am = oo[f'blocks.{len(oo)-1}.attn.softmax'].cpu().numpy()[0]
            print(am.shape)
    else:
        pred = []
        am = None

    n = 0
    for i, ((x1, y1, x2, y2), orig_label, image_path) in enumerate(predictions):
        if orig_label in ["human", "vehicle"]:
            table_data.append([i, orig_label, orig_label, image_path])
        else:
            top_class = max({classes[j]: float(pred[n][j]) for j in range(len(classes))}, key=lambda k: pred[n][classes.index(k)])
            top_score = float(pred[n][classes.index(top_class)])*100
            seq_label = f"{top_class}: {top_score:.0f}%"
            table_data.append([i, orig_label, seq_label, image_path])
            n += 1

    # permutate bbox order to make the attention matrix easier to read
    sp = [table_data[i][2].split(":")[0] for i in range(len(table_data))]
    perm = sorted(range(len(sp)), key=lambda k: sp[k])
    gallery_images = [gallery_images[i] for i in perm]
    img_list = [img_list[i] for i in perm]
    table_data = [table_data[i] for i in perm]
    if am is not None:
        not_animal = [x in ["human", "vehicle"] for x in [sp[i] for i in perm]]
        not_animal_idx = [perm[i] for i in range(len(perm)) if not_animal[i]]
        am_perm = []
        for i in range(len(perm)):
            if not_animal[i]:
                continue
            else:
                am_perm.append(perm[i] - sum([perm[i] > k for k in not_animal_idx]))
        am = am[:, am_perm, :]
        am = am[:, :, am_perm]
        img_list = [img_list[i] for i in range(len(img_list)) if not not_animal[i]]

    # annotate bounding boxes on images
    annotated_image1 = cv2.imread(image1) if image1 else None
    annotated_image2 = cv2.imread(image2) if image2 else None
    for i, ((x1, y1, x2, y2), orig_label, image_path) in enumerate(predictions):
        if image_path == image1:
            annotate_bbox_on_image(annotated_image1, x1, y1, x2, y2, str(perm.index(i)+1))
        else:
            annotate_bbox_on_image(annotated_image2, x1, y1, x2, y2, str(perm.index(i)+1))

    gallery_images = make_thumbnail_gallery(gallery_images, table_data)

    # make attention matrix
    img = np.concatenate(img_list, axis=1)
    X, Y, _ = img.shape
    am_img = np.zeros((X+Y, X+Y, 3), dtype=np.uint8)
    if am is not None:
        am_img[:X, X:X+Y] = img
        am_img[X:X+Y, :X] = np.concatenate(img_list, axis=0)
        am_img[X:, X:] = cv2.resize(get_sns_heatmap(am), (Y, Y), interpolation=cv2.INTER_NEAREST)

    output_image1 = cv2.cvtColor(annotated_image1, cv2.COLOR_BGR2RGB) if image1 else None
    output_image2 = cv2.cvtColor(annotated_image2, cv2.COLOR_BGR2RGB) if image2 else None
    return output_image1, output_image2, gallery_images, am_img


with gr.Blocks(css=".block {margin: auto}") as demo:
    args = dict(show_download_button=False, show_fullscreen_button=False) if False else dict()
    with gr.Tab("One image"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Image")
                path = [x for x in glob(os.path.join(dirname, '../images/gradio_examples/one_image/*')) if x.lower().split(".")[-1] in ["jpg", "jpeg", "png"]]
                examples = gr.Examples(path, input_image)
                btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=3):
                with gr.Group():
                    with gr.Row():
                        output_image = gr.Image(type="numpy", label="Annotated Image", scale=16, show_label=False, **args)
                        output_image2 = gr.Image(type="numpy", label="Annotated Image 2", scale=0, render=False, show_label=False, **args)
                        am_image = gr.Image(type="numpy", label="Attention Map", scale=9, show_label=False, **args)
                    with gr.Row():
                        output_gallery = gr.Gallery(label="Detected Crops", columns=9, rows=2, height="auto", object_fit='scale-down', show_label=False)
        btn.click(
            fn=show_preds_image,
            inputs=[input_image],
            outputs=[output_image, output_image2, output_gallery, am_image]
        )
    with gr.Tab("Two images"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Image")
                path2 = [x for x in glob(os.path.join(dirname, '../images/gradio_examples/two_images/*')) if x.lower().split(".")[-1] in ["jpg", "jpeg", "png"]]
                path2.sort()
                examples = gr.Examples(path2, input_image)
                input_image2 = gr.Image(type="filepath", label="Second Image (optional)")
                examples2 = gr.Examples(path2, input_image2)
                btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=3):
                with gr.Group():
                    with gr.Row():
                        output_image = gr.Image(type="numpy", label="Annotated Image", scale=4, show_label=False, **args)
                        output_image2 = gr.Image(type="numpy", label="Annotated Image 2", scale=4, show_label=False, **args)
                        am_image = gr.Image(type="numpy", label="Attention Map", scale=3, show_label=False, **args)
                    with gr.Row():
                        output_gallery = gr.Gallery(label="Detected Crops", columns=9, rows=2, height="auto", object_fit='scale-down', show_label=False)
        btn.click(
            fn=show_preds_image,
            inputs=[input_image, input_image2],
            outputs=[output_image, output_image2, output_gallery, am_image]
        )

demo.launch(share=False, allowed_paths=[os.path.join(dirname, '../images/gradio_examples/')])
