# Vehicle Detection and Tracking

This repository contains the implementation developed for the research paper:

**"Lightweight YOLO Frameworks for Vehicle Detection and Re-ID-Enhanced DeepSORT Tracking"**

The project evaluates several lightweight YOLO-based object detectors combined with an enhanced DeepSORT tracking framework for vehicle detection and multi-object tracking.

The system is trained and evaluated primarily on the **UA-DETRAC dataset**, with an additional **Re-Identification (ReID)** training stage based on the **VeRI-776 dataset**.

---

# Project Overview

The pipeline implemented in this project follows these main stages:

1. Dataset acquisition  
2. Dataset preprocessing  
3. Dataset organization and deployment preparation  
4. YOLO model training  
5. Detection evaluation  
6. ReID model training for DeepSORT  
7. Detection + tracking integration  
8. Tracking evaluation  
9. Real-time visualization and traffic report generation

Each stage is described below.

---

# 1. Dataset Acquisition

The primary dataset used in this project is **UA-DETRAC**, a large-scale benchmark for vehicle detection and multi-object tracking.

At the time of writing, the dataset is not publicly hosted online. To obtain it, users typically need to request access from researchers associated with the **University at Albany (SUNY)** or other researchers who maintain a local copy.

---

# 2. Dataset Preprocessing

Before training, the dataset must be converted into a format compatible with YOLO detectors.

The preprocessing pipeline includes:

1. **Annotation conversion**  
   - UA-DETRAC annotations are converted to the **YOLO bounding box format**.

2. **Video frame extraction**  
   - All videos are decomposed into individual frames.

3. **Frame subsampling**  
   - To reduce dataset redundancy and improve training efficiency, only **one frame out of every ten frames** is retained.  
   - This reduces temporal similarity between samples and helps mitigate potential overfitting.

---

# 3. Dataset Organization

After preprocessing, the dataset is uploaded to **Roboflow** to simplify dataset management and deployment.

Roboflow allows for:
- easier dataset versioning
- annotation visualization
- direct export to YOLO-compatible formats

The dataset used in this project can be accessed here:

https://universe.roboflow.com/altice-lzmaq/vehicle-detection-and-tracking-l2epo

For reproducibility, it is **strongly recommended to start from this dataset version**.

---

# 4. YOLO Model Training

Several **lightweight YOLO architectures** were trained and benchmarked on the processed dataset.

Multiple training configurations were explored in order to:

- optimize model performance
- prevent GPU memory overflow
- avoid training crashes caused by large model variants

During training, the **best-performing checkpoint (`best.pt`)** is automatically saved for each experiment.

---

# 5. Detection Evaluation

After training, the detectors are evaluated on **40 unseen test videos** from the UA-DETRAC dataset.

The test videos must undergo the **same preprocessing steps** applied to the training data:

- annotation conversion
- frame extraction
- frame subsampling

From the inference stage, several detection metrics are computed, including:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- Training loss values

These metrics provide insights into the detection performance of each YOLO architecture.

---

# 6. ReID Model Training (DeepSORT)

Before running the tracking pipeline, the **Re-Identification (ReID) module used by DeepSORT** must be trained.

This project uses the **VeRI-776 dataset**, which can be obtained by contacting the **Beijing University of Posts and Telecommunications**.

The ReID backbone is retrained using this dataset to produce two variants:

- **Standard ReID module**
- **Enhanced ReID module**

This allows a direct comparison between the baseline DeepSORT pipeline and the improved version proposed in the research.

---

# 7. Detection and Tracking Integration

All trained YOLO detectors are integrated with the DeepSORT tracking framework.

Each combination of:

- YOLO detector
- DeepSORT variant

is evaluated on the **40 unseen test videos**.

Detection reports are generated for **10 different confidence thresholds**, allowing a detailed analysis of system performance across different detection sensitivities.

---

# 8. Tracking Evaluation

The generated detection and tracking results are processed using the **UA-DETRAC evaluation toolkit**, which was obtained by the authors of the UA-DETRAC dataset.

This toolkit computes both detection and tracking metrics, including:

Detection metric:
- AP@0.7

Tracking metrics:
- MOTA
- MOTP
- ID Switches
- Mostly Tracked (MT)
- Mostly Lost (ML)
- Fragmentation (FM)

Evaluation is performed within a **region of interest (ROI)** defined for each video sequence.

---

# 9. Real-Time Visualization

A final experiment allows the system to perform **real-time vehicle detection and tracking**.

During this stage, the system:

- visually displays detected vehicles
- assigns consistent IDs to tracked vehicles
- generates a report summarizing vehicle traffic activity

This stage helps qualitatively assess the pipeline in a realistic deployment scenario.

---

# Contact

For questions regarding the project or implementation details, please contact:

**João Matias**  
joaomatiasgoncalves321@hotmail.com
