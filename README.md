# Vehicle Detection and Tracking

A lightweight computer-vision research framework for vehicle detection and multi-object tracking in smart-city traffic footage.

The project combines lightweight YOLO detectors with DeepSORT and explores appearance-based vehicle re-identification (Re-ID).

## Pipeline

1. Prepare vehicle detection data.
2. Train or evaluate a lightweight YOLO detector.
3. Track detected vehicles using DeepSORT.
4. Evaluate detection and tracking performance.
5. Run a video demonstration with tracked vehicles and a summary of observed activity.

## Main files

- `run.py` — YOLO training example.
- `tracker.py` — tracking integration.
- `deepsort_inference.py` — detection and tracking workflow.
- `show_video_updated_report.py` — video demonstration and traffic summary.
- `cut_frame.py`, `remap_labels.py` — data preparation utilities.
- `deep_sort/` — DeepSORT implementation and supporting utilities.
- `data.yaml` — example dataset configuration.

## Datasets

Experiments used **UA-DETRAC** for vehicle detection and tracking, and **VeRi-776** for vehicle re-identification. The datasets are not included; obtain them from their official sources and follow their terms of use.

A Roboflow dataset export was also used during development:
https://universe.roboflow.com/altice-lzmaq/vehicle-detection-and-tracking-l2epo

Check its version, annotations, and licence before reusing it.

## Setup

Use Python and install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Before running an example, configure the dataset and model paths for your environment. Required model weights are not included in the public repository. Dependency versions and compatibility may need adjustment.

## Evaluation

Depending on the experiment, evaluation includes precision, recall, mean average precision (mAP), Multiple Object Tracking Accuracy (MOTA), and identity switches.

## Scope

This repository presents research code and a demonstration pipeline; it is not a claim of a fully deployed traffic-monitoring service.

## Contact

João Matias
