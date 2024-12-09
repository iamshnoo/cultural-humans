"""
# Python 3.8.6
pip install torchmetrics transformers tqdm
"""
import os
import pandas as pd
import numpy as np
import argparse
import torch
import gc
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from torchmetrics.functional.multimodal import clip_score
import PIL
from PIL import Image
import torch.nn.functional as F


# Function to load and preprocess an image
def load_image(path):
    try:
        image = PIL.Image.open(path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None


# Function to calculate cosine similarity between two images
def calculate_image_similarity(
    source_image_path, target_image_path, model, processor, device
):
    source_image = load_image(source_image_path)
    target_image = load_image(target_image_path)

    if source_image is None or target_image is None:
        return None

    # Preprocess images
    source_input = processor(source_image, return_tensors="pt").to(device)
    target_input = processor(target_image, return_tensors="pt").to(device)

    # Extract features
    with torch.no_grad():
        source_features = model(**source_input).last_hidden_state.mean(dim=1)
        target_features = model(**target_input).last_hidden_state.mean(dim=1)

    # Normalize features
    source_features = F.normalize(source_features, p=2, dim=1)
    target_features = F.normalize(target_features, p=2, dim=1)

    # Cosine similarity
    cosine_similarity = F.cosine_similarity(source_features, target_features).item()

    # Clear cache to prevent memory issues
    torch.cuda.empty_cache()
    gc.collect()

    return cosine_similarity


# Function to calculate CLIP score deltas
def compute_clip_score_deltas(
    source_image_path, target_image_path, src_country, tgt_country, device
):
    source_image = load_image(source_image_path)
    target_image = load_image(target_image_path)

    if source_image is None or target_image is None:
        return None, None

    # Resize images to 224x224 for CLIP
    source_resized = source_image.resize((224, 224))
    target_resized = target_image.resize((224, 224))

    # Convert images to tensors
    source_image_tensor = (
        torch.from_numpy(np.array(source_resized))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )
    target_image_tensor = (
        torch.from_numpy(np.array(target_resized))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # Compute CLIP scores
    with torch.no_grad():
        score_src_image_src_country = clip_score(
            source_image_tensor,
            src_country,
            model_name_or_path="openai/clip-vit-base-patch16",
        ).item()
        score_src_image_tgt_country = clip_score(
            source_image_tensor,
            tgt_country,
            model_name_or_path="openai/clip-vit-base-patch16",
        ).item()
        score_target_image_src_country = clip_score(
            target_image_tensor,
            src_country,
            model_name_or_path="openai/clip-vit-base-patch16",
        ).item()
        score_target_image_tgt_country = clip_score(
            target_image_tensor,
            tgt_country,
            model_name_or_path="openai/clip-vit-base-patch16",
        ).item()

    delta1 = score_target_image_src_country - score_src_image_src_country
    delta2 = score_target_image_tgt_country - score_src_image_tgt_country

    # Clear cache to prevent memory issues
    torch.cuda.empty_cache()
    gc.collect()

    return delta1, delta2


# Function to create metadata file
def create_metadata(base_dir="../results", output_csv="../results/metadata.csv"):
    data = []
    for setting in ["setting1", "setting2"]:
        setting_path = os.path.join(base_dir, setting)
        for src_tgt_folder in os.listdir(setting_path):
            if "_to_" not in src_tgt_folder:
                continue
            src_country, tgt_country = src_tgt_folder.split("_to_")
            target_path = os.path.join(setting_path, src_tgt_folder)
            for image_file in os.listdir(target_path):
                activity = image_file.split(".")[0]
                src_path = os.path.join(
                    "../images", setting, f"{src_country}_{activity}.png"
                )
                target_path = os.path.join(setting_path, src_tgt_folder, image_file)
                data.append(
                    {
                        "src_path": src_path,
                        "target_path": target_path,
                        "src_country": src_country,
                        "tgt_country": tgt_country,
                        "setting": setting,
                        "activity": activity,
                    }
                )

    metadata_df = pd.DataFrame(data)
    metadata_df.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")


# Function to evaluate images based on filtered metadata
def evaluate_filtered_metadata(
    metadata_csv,
    src_country,
    setting,
    output_csv,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    metadata_df = pd.read_csv(metadata_csv)
    filtered_data = metadata_df[
        (metadata_df["src_country"] == src_country)
        & (metadata_df["setting"] == f"setting{setting}")
    ]

    image_processor = ViTImageProcessor.from_pretrained(
        "facebook/dino-vitb8", cache_dir="../models/dino-vit"
    )
    image_model = (
        ViTModel.from_pretrained("facebook/dino-vitb8", cache_dir="../models/dino-vit")
        .eval()
        .to(device)
    )

    results = []
    for _, row in tqdm(filtered_data.iterrows(), total=len(filtered_data)):
        try:
            similarity = calculate_image_similarity(
                row["src_path"],
                row["target_path"],
                image_model,
                image_processor,
                device,
            )
            delta1, delta2 = compute_clip_score_deltas(
                row["src_path"],
                row["target_path"],
                row["src_country"],
                row["tgt_country"],
                device,
            )

            results.append(
                {
                    "src_path": row["src_path"],
                    "target_path": row["target_path"],
                    "src_country": row["src_country"],
                    "tgt_country": row["tgt_country"],
                    "setting": row["setting"],
                    "activity": row["activity"],
                    "similarity": similarity,
                    "delta1": delta1,
                    "delta2": delta2,
                }
            )
        except Exception as e:
            print(f"Error processing row {row}: {e}")

        torch.cuda.empty_cache()
        gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Evaluation results saved to {output_csv}")


# Main Workflow
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Source country")
    parser.add_argument(
        "--setting", type=str, default="1", help="Image setting (1 or 2)"
    )
    parser.add_argument(
        "--create_metadata",
        action="store_true",
        help="Create metadata file before evaluation",
    )
    args = parser.parse_args()

    if args.src == "South_Africa":
        args.src = "South Africa"
    if args.src == "United_States":
        args.src = "United States"
    if args.src == "United_Kingdom":
        args.src = "United Kingdom"

    metadata_csv = "../results/metadata.csv"
    output_csv = f"../results/metrics/setting{args.setting}/{args.src}_setting{args.setting}_metrics.csv"
    print("output path:", output_csv)

    if args.create_metadata:
        create_metadata(base_dir="../results", output_csv=metadata_csv)

    evaluate_filtered_metadata(
        metadata_csv, str(args.src), str(args.setting), output_csv
    )
