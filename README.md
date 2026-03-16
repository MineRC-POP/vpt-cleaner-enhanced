# VPT-Cleaner-Enhanced

A reproduction and enhancement project for the data cleaning recognition model used in the original VPT pipeline.

> This repository aims to reproduce the data cleaning / recognition stage of the original VPT paper and further enhance it for practical dataset construction.

---

## Overview

Large-scale Minecraft video pretraining pipelines such as VPT rely heavily on high-quality video data.  
However, real-world web videos often contain many unusable or harmful frames, such as:

- watermark logos
- facecam overlays
- text overlays / subtitles
- platform UI overlays
- non-survival gameplay
- non-Minecraft frames
- inventory / container / loading / other UI screens

This project is built to address that problem.

**VPT-Cleaner-Enhanced** is a multi-task image recognition project designed to identify and clean Minecraft video frames.  
It reproduces the spirit of the original VPT data cleaning model and extends it with a more complete engineering workflow, including:

- video search
- batch download
- frame extraction
- manual annotation
- model training
- inference
- AI-assisted labeling
- cleaned frame export

The training set contains **8000 manually annotated images**, and **all labels are created by hand**.

---

## Features

- Search candidate Minecraft survival videos from YouTube
- Batch download **720p no-audio** videos
- Extract frames from downloaded videos
- Manually annotate frames with a GUI tool
- Train a multi-task cleaner model
- Run single-image inference
- Use the trained model for AI-assisted annotation
- Export cleaned / usable frames from annotation results

---

## Tasks

The model predicts three categories of information for each frame:

### 1. Game Mode
- `survival`
- `non_survival`
- `not_minecraft`
- `unknown_mode`

### 2. Pollution Types
- `watermark_logo`
- `facecam_person`
- `text_overlay`
- `platform_ui_overlay`
- `border_frame`
- `other_artifact`

### 3. UI Type
- `none`
- `chat`
- `pause_menu`
- `settings`
- `inventory`
- `container`
- `death_screen`
- `loading`
- `other_ui`

---

## Repository Structure

```text
.
├─ search.py
├─ download_from_jsonl.py
├─ extract_frames.py
├─ cleaner_labeler.py
├─ train_cleaner_multitask.py
├─ val.py
├─ clean_original.py
├─ cookies.txt               # ignored, local-only
├─ frames/                   # extracted frames
├─ frames_original/          # original extracted frames backup / source frames
├─ frame_true/               # cleaned / accepted frames
├─ runs/                     # training outputs and checkpoints
├─ videos/                   # downloaded videos
├─ videos_original/          # original video backups
├─ yt_dataset_out/
│  ├─ meta/
│  ├─ transcripts/
│  └─ videos/
└─ __pycache__/
````

---

## File Description

### `search.py`

Searches candidate Minecraft videos from YouTube and stores metadata locally.

Main responsibilities:

* search Minecraft survival / longplay videos
* support YouTube API or `yt-dlp`-based collection
* filter by blacklist keywords
* collect video metadata
* collect subtitles / transcript-related information when available
* save results into dataset metadata files

---

### `download_from_jsonl.py`

Reads the collected metadata file and downloads videos in batch.

Main responsibilities:

* read video entries from JSONL metadata
* batch download videos into local folders
* prefer **720p no-audio** format
* skip files that already exist
* support local `cookies.txt` when needed

---

### `extract_frames.py`

Extracts image frames from downloaded videos.

Main responsibilities:

* scan videos from the local video directory
* use `ffprobe` to get video duration
* allocate extraction counts by video duration ratio
* call `ffmpeg` to export frames
* generate image datasets for manual labeling and training

---

### `cleaner_labeler.py`

A GUI annotation tool for manually labeling frames.

Main responsibilities:

* browse frame images
* annotate:

  * game mode
  * pollution types
  * UI type
  * uncertainty
* save labels into `annotations.json`
* support keyboard shortcuts
* support AI-assisted labeling by loading a trained model

This is one of the most important tools in the whole project.

---

### `train_cleaner_multitask.py`

Trains the multi-task cleaner model.

Main responsibilities:

* load images and annotations
* build training / validation splits
* train a shared backbone with multiple heads
* output checkpoints and validation results

The current training design uses a shared CNN backbone with three output heads for:

* game mode classification
* pollution multi-label classification
* UI type classification

---

### `val.py`

Runs inference and validation for trained models.

Main responsibilities:

* load model checkpoints
* run prediction on a single image
* output predicted game mode
* output predicted UI type
* output predicted pollution labels
* optionally export results as JSON

It is also used by the annotation tool for AI-assisted labeling.

---

### `clean_original.py`

Exports or copies frames that satisfy cleaning conditions.

Main responsibilities:

* read annotation results
* locate source images
* copy qualified frames into `frame_true/`
* help produce a cleaned image subset for later use

---

## Workflow

A typical workflow looks like this:

### 1. Search videos

Use `search.py` to collect candidate Minecraft videos from YouTube.

### 2. Download videos

Use `download_from_jsonl.py` to batch download 720p no-audio videos.

### 3. Extract frames

Use `extract_frames.py` to convert videos into image frames.

### 4. Annotate data

Use `cleaner_labeler.py` to manually label the frames.

### 5. Train model

Use `train_cleaner_multitask.py` to train the cleaner model.

### 6. Inference / assisted labeling

Use `val.py` or the built-in model-loading feature in `cleaner_labeler.py`.

### 7. Export cleaned dataset

Use `clean_original.py` to export acceptable frames.

---

## Dataset

This project is trained on:

* **8000 images**
* **100% manually annotated**
* collected from Minecraft-related video frames
* designed for VPT-style data cleaning and recognition tasks

### Dataset Availability

**The dataset is NOT open-sourced due to copyright restrictions.**

The original frame data is derived from online video content, and redistributing the dataset may introduce copyright issues.
Therefore:

* the **dataset will not be released**
* the **training code is public**
* the **trained model weights are provided in Releases**

Please check the repository **Releases** page for available model files.

---

## Model

The project uses a multi-task image recognition model to classify frame usability and contamination.

Current design:

* shared backbone
* game mode head
* pollution head
* UI type head

This repository is intended to reproduce the original VPT cleaner logic and improve it with a more usable engineering pipeline.

---

## Project Goal

This repository is **not** intended to be an exact line-by-line reimplementation of the original VPT paper.
Instead, it aims to:

1. reproduce the **core idea** of the VPT data cleaning recognition model
2. rebuild a usable engineering pipeline around it
3. enhance the original approach for practical annotation and dataset construction

In other words, this is both:

* a **reproduction project**
* and an **enhanced data cleaning toolkit**

---

## Release

Pretrained model weights are published in **Releases**.

You can download them from the repository release page and use them for:

* validation
* inference
* AI-assisted annotation

---

## Copyright and License

Copyright © 2026 Evidence.
**All rights reserved.**

This repository is provided for research, learning, and reference purposes only.

### Notes

* The source code in this repository is copyrighted.
* The dataset is not distributed.
* Pretrained weights may be distributed separately through Releases.
* Without explicit permission, redistribution of the dataset is not allowed.

At the current stage, this project is released as **source-visible, not open-source licensed**.

---

## TODO

* [ ] Improve classification accuracy on hard negative samples
* [ ] Add more pollution categories
* [ ] Support temporal / multi-frame modeling
* [ ] Add active learning for efficient annotation
* [ ] Improve AI-assisted labeling workflow
* [ ] Provide better evaluation scripts
* [ ] Add confusion matrix and per-class metrics export

---

## Acknowledgment

This project is inspired by the data processing idea behind the original VPT paper and is intended to reproduce and enhance its data cleaning recognition stage for Minecraft video datasets.

---

## Contact

If you are interested in this project, please open an issue or check the Releases page for available model weights.
