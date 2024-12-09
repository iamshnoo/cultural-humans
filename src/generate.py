"""
# Python 3.8.6
pip install torch
pip install transformers
pip install numpy pandas tqdm requests opencv-python
pip install diffusers --upgrade
"""
import os
import cv2
import json
import numpy as np
import pandas as pd
import requests
from torch import autocast
import argparse
import torch
import random
from tqdm import tqdm
from diffusers import DiffusionPipeline


# Parse arguments
parser = argparse.ArgumentParser(description="Generate images using Open Source model.")
parser.add_argument(
    "--setting", type=str, default="1", help="Setting of images to generate"
)
# Add test_type argument here
parser.add_argument(
    "--test_type",
    type=str,
    default="sampletest",
    help="Type of test: randomtest or sampletest",
)
args = parser.parse_args()  # Ensure arguments are parsed correctly

# Access the arguments
setting = str(args.setting)
test_type = str(args.test_type)  # Now test_type should be accessible

# Load activities data
with open("activities.json", "r") as f:
    data = json.load(f)
    activities = data["Activities"]


# Function to get a random activity
def get_random_activity():
    return random.choice(activities)


# Select activities based on the test type
if test_type == "randomtest":
    # Randomly select 10 unique activities
    random_activities = []
    while len(random_activities) < 10:
        activity = get_random_activity()
        if activity not in random_activities:
            random_activities.append(activity)

elif test_type == "sampletest":
    # Use all activities in the dataset
    random_activities = activities[:]

# Load country names from the JSON file
with open("countries.json", "r") as file:
    country_data = json.load(file)

# Extract the list of countries
countries = country_data.get("Countries", [])

pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    cache_dir="../models/flux",
    torch_dtype=torch.float16,
)
# Enable attention slicing
pipeline.enable_attention_slicing()

# Move the pipeline to CUDA
pipeline.to("cuda")


# Function to generate and save images
def generate_and_save_images(prompt, idx, activity_desc, country, setting):
    # Determine the directory based on the setting
    directory = f"../images/setting{setting}/"
    os.makedirs(directory, exist_ok=True)

    try:
        with autocast("cuda"):
            generator = torch.Generator("cuda").manual_seed(random.randint(0, 1000000))
            image = pipeline(
                prompt, generator=generator, num_inference_steps=20
            ).images[0]
        # Sanitize activity description for filenames
        sanitized_activity = (
            activity_desc.replace(" ", "_") if activity_desc else "generic"
        )
        image_path = os.path.join(directory, f"{country}_{sanitized_activity}.png")

        # Save the image
        image.save(image_path)
        print(f"Saved: {image_path}")
        return image_path
    except Exception as e:
        print(f"Error generating image for prompt {prompt}: {str(e)}")
        return None


# Prepare prompts based on the setting
prompts = []

if setting == "1":
    if test_type == "sampletest":  # Use all activities and all countries
        for activity in random_activities:  # Loop over all activities
            for country in countries:  # Loop over all countries
                prompts.append(
                    (
                        f"A person from {country} engaged in {activity}, with their face visible.",
                        # f"A person from {country}, engaged in {activity}, facing front, with their entire body visible, and their face clearly visible.",
                        activity,
                        country,
                    )
                )
    else:  # Use a random sample for "randomtest"
        for activity in random.sample(
            random_activities, 5
        ):  # Randomly sample 5 activities
            for country in random.sample(countries, 1):  # Randomly sample 5 countries
                prompts.append(
                    (
                        f"A person from {country} engaged in {activity}, with their face visible.",
                        # f"A person from {country}, engaged in {activity}, facing front, with their entire body visible, and their face clearly visible.",
                        activity,
                        country,
                    )
                )

elif setting == "2":
    if test_type == "sampletest":  # Use all countries
        for country in countries:  # Loop over all countries
            prompts.append(
                (
                    f"A person from {country}, wearing culturally relevant clothing, facing front, with their face clearly visible.",
                    # f"A person from {country}, wearing culturally relevant clothing, facing front, with their entire body visible, and their face clearly visible.",
                    None,  # No activity description
                    country,
                )
            )
    else:  # Use a random sample for "randomtest"
        for country in random.sample(countries, 5):  # Randomly sample 5 countries
            prompts.append(
                (
                    f"A person from {country}, wearing culturally relevant clothing, facing front, with their face clearly visible.",
                    # f"A person from {country}, wearing culturally relevant clothing, facing front, with their entire body visible, and their face clearly visible.",
                    None,  # No activity description
                    country,
                )
            )


# Shuffle the final list of prompts to ensure randomness
random.shuffle(prompts)

# Generate images for each prompt
for idx, (prompt, activity_desc, country) in enumerate(
    tqdm(prompts, total=len(prompts))
):
    generate_and_save_images(prompt, idx, activity_desc, country, setting)
