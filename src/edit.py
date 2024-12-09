"""
# Python 3.8.6
pip install torch torchvision
pip install huggingface_hub
pip install transformers
pip install diffusers --upgrade
pip install git+https://github.com/xinyu1205/recognize-anything.git
pip install transformers --upgrade
pip install segment_anything
pip install ultralytics
mkdir models && cd models
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download
from ram.models import ram_plus
from ram import get_transform
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    ViTFeatureExtractor,
    ViTForImageClassification,
)
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from ultralytics import YOLO
import torch
import argparse
import json
from itertools import permutations


# Initialize Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load RAM+ Model for open set Tag Detection
image_size = 384
transform = get_transform(image_size=image_size)
ram_plus_model_path = hf_hub_download(
    repo_id="xinyu1205/recognize-anything-plus-model",
    filename="ram_plus_swin_large_14m.pth",
    cache_dir="../models/ram_plus",
)
ram_model = ram_plus(
    pretrained=ram_plus_model_path, image_size=image_size, vit="swin_l"
)
ram_model.eval()
ram_model = ram_model.to(device)

# Load Grounding DINO Model for object bounding boxes to ground tags and feed to SAM
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id, cache_dir="../models/edits")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id, cache_dir="../models/edits"
).to(device)

# Load SAM Model for segmentation of non-face regions
# first need to do wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam_checkpoint = "../models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# Load YOLOv8 for Face Detection
yolo_model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt",
    cache_dir="../models/yolov8-face-detection",
)
yolo_model = YOLO(yolo_model_path)

# Load Age and Gender Models from FairFace
age_model = ViTForImageClassification.from_pretrained(
    "nateraw/vit-age-classifier",
    cache_dir="../models/vit-age-classifier",
).to(device)
age_transforms = ViTFeatureExtractor.from_pretrained(
    "nateraw/vit-age-classifier",
    cache_dir="../models/vit-age-classifier",
)

gender_model = ViTForImageClassification.from_pretrained(
    "dima806/fairface_gender_image_detection",
    cache_dir="../models/vit-age-classifier",
).to(device)
gender_transforms = ViTFeatureExtractor.from_pretrained(
    "dima806/fairface_gender_image_detection",
    cache_dir="../models/vit-age-classifier",
)

# Load Stable Diffusion Inpainting Pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    cache_dir="../models/stable-diffusion",
).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# Helper Functions
def detect_tags(image):
    """Detect tags using RAM+."""
    transformed_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        tags = ram_model.generate_tag_openset(transformed_image)
    tags = [item.strip() for item in tags[0].split("|")]
    tags_string = ". ".join(tags) + "."
    return tags, tags_string


def detect_faces(image):
    """Detect faces using YOLOv8."""
    results = yolo_model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes
    return detections


def create_mask(image, boxes):
    """Create a mask for detected face areas."""
    mask = Image.new("L", image.size, 0)
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                mask.putpixel((x, y), 255)
    return mask


def get_age_gender(face_image):
    """Predict age and gender for a cropped face."""
    inputs = age_transforms(face_image, return_tensors="pt").to(device)
    age_output = age_model(**inputs)
    age_proba = age_output.logits.softmax(1)
    age_preds = age_proba.argmax(1)
    age_label = age_model.config.id2label[age_preds.item()]

    gender_inputs = gender_transforms(face_image, return_tensors="pt").to(device)
    gender_output = gender_model(**gender_inputs)
    gender_proba = gender_output.logits.softmax(1)
    gender_preds = gender_proba.argmax(1)
    gender_label = gender_model.config.id2label[gender_preds.item()]

    return age_label, gender_label


def get_grounding_boxes(image, tags_string):
    """Get grounding DINO bounding boxes for non-face regions."""
    inputs = processor(images=image, text=tags_string, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"]
    return boxes, labels


def generate_sam_masks_with_boxes(image, boxes):
    """Generate SAM masks using bounding boxes."""
    image_np = np.array(image)
    predictor.set_image(image_np)

    masks = []
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        input_box = np.array([[xmin, ymin, xmax, ymax]])
        mask, _, _ = predictor.predict(box=input_box, multimask_output=False)
        masks.append(mask[0])  # Take the first mask
    return masks


def inpaint_faces(image, face_mask, src_country, tgt_country, face_boxes):
    """Inpaint face regions with age and gender."""
    for box in face_boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cropped_face = image.crop((xmin, ymin, xmax, ymax))
        age, gender = get_age_gender(cropped_face)
        prompt = f"Face of a {gender.lower()} of age {age} years from {tgt_country}."
        print(f"Face Inpainting Prompt: {prompt}")

        generator = torch.Generator(device).manual_seed(random.randint(0, 100000))
        image_resized = image.resize((512, 512))
        mask_resized = face_mask.resize((512, 512))

        inpainted_face = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, unrealistic",
            image=image_resized,
            mask_image=mask_resized,
            generator=generator,
        ).images[0]

        image = inpainted_face.resize(image.size)
    return image


def inpaint_non_faces(image, masks, labels, target_country, face_mask):
    """Inpaint non-face regions iteratively."""
    current_image = np.array(image)
    face_mask_np = np.array(face_mask) > 0  # Binary face mask

    for idx, mask in enumerate(masks):
        label = labels[idx]
        prompt = f"{target_country}. {label}. intricate details. high quality. photorealistic."
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_blurred = mask_image.filter(ImageFilter.GaussianBlur(15))
        image_resized = Image.fromarray(current_image).resize((512, 512))
        mask_resized = mask_blurred.resize((512, 512))

        generator = torch.Generator(device).manual_seed(random.randint(0, 100000))
        inpainted_image = pipe(
            prompt=prompt,
            negative_prompt="ugly, blurry, low res, unrealistic",
            image=image_resized,
            mask_image=mask_resized,
            generator=generator,
        ).images[0]

        inpainted_np = np.array(inpainted_image.resize(image.size))

        mask_np = np.array(mask_blurred) / 255.0
        current_image = (
            mask_np[:, :, None] * inpainted_np
            + (1 - mask_np[:, :, None]) * current_image
        ).astype(np.uint8)

    return Image.fromarray(current_image)


# Main Workflow
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting", type=str, default="1", help="Setting of images to edit"
    )
    parser.add_argument("--src", type=str, default="Algeria", help="Source country")
    args = parser.parse_args()

    setting = str(args.setting)
    src_country = str(args.src)

    if src_country == "South_Africa":
        src_country = "South Africa"
    if src_country == "United_States":
        src_country = "United States"
    if src_country == "United_Kingdom":
        src_country = "United Kingdom"

    with open("countries.json", "r") as f:
        countries_data = json.load(f)
    countries = countries_data["Countries"]
    targets = [country for country in countries if country != src_country]
    targets.sort()

    for target_country in targets:
        if setting == "1":
            src_path = "../images/setting1/"
            target_path = f"../results/setting1/{src_country}_to_{target_country}/"
        elif setting == "2":
            src_path = "../images/setting2/"
            target_path = f"../results/setting2/{src_country}_to_{target_country}/"

        images = [f for f in os.listdir(src_path) if f.startswith(src_country)]
        for image_path in images:
            activity = image_path.split(src_country + "_")[1].split(".")[0]
            # editing steps
            image = Image.open(src_path + image_path).convert("RGB")
            face_boxes = detect_faces(image)
            face_mask = create_mask(image, face_boxes)
            tags, tags_string = detect_tags(image)
            grounding_boxes, labels = get_grounding_boxes(image, tags_string)
            masks = generate_sam_masks_with_boxes(image, boxes=grounding_boxes)
            inpainted_face_image = inpaint_faces(
                image, face_mask, src_country, target_country, face_boxes
            )
            final_image = inpaint_non_faces(
                inpainted_face_image, masks, labels, target_country, face_mask
            )
            # save in target path
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            final_image.save(target_path + activity + ".png")

            print("-" * 80)
            print("Setting:", setting)
            print("Source Country:", src_country)
            print("Target Country:", target_country)
            print("Activity:", activity)
            print(f"Saved: {target_path + activity + '.png'}")
            print("-" * 80)
