# Cultural Image Translation with a focus on Human Figures in Images

This project aims to translate images from one culture to another
while preserving the human figures in the images, by applying several computer
vision techniques like face detection, open-set object detection, object
grounding using bounding boxes, segmentation masking and image inpainting.
All components are open-source and do not require any API keys.

## Folder structure

1. Run generate.py to create the data in the images folder.
2. Run edit.py to create edits in the results/setting1 and results/setting2 folders.
3. Run evals.py to evaluate the edits and store metrics in the results/metrics folder.
4. Run analysis.py to compile all metrics and create a summarized results table.
5. Run plot.py to create plots from the summarized results.

```bash
.
└── images
    ├── setting1/
    ├── setting2/
└── models
└── results
    ├── figs/
    ├── metrics/
    ├── setting1/
    ├── setting2/
    ├── metadata.csv
    ├── metrics.csv
    ├── summary_table.csv
└── src
    ├── activities.json
    ├── analysis.py
    ├── countries.json
    ├── edit.py
    ├── evals.py
    ├── generate.py
    ├── run_array.slurm
    └── splits.txt
```

## Requirements

```bash
Python 3.8.6
```

```bash
pip install numpy pandas seaborn matplotlib tqdm
pip install torch torchvision
pip install torchmetrics
pip install huggingface_hub
pip install transformers --upgrade
pip install diffusers --upgrade
pip install git+https://github.com/xinyu1205/recognize-anything.git
pip install transformers --upgrade
pip install segment_anything
pip install ultralytics
mkdir models && cd models
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
