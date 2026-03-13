# Vehicle-Detection-and-Tracking

This entire project was developed on the context of a paper entitled "Lightweight YOLO Frameworks for Vehicle Detection and Re-ID-Enhanced DeepSORT Tracking".

For this project, several YOLO detectors were trained and benchmarked on the UA-DETRAC dataset together with an enhanced DeepSORT approach.

To start, it's first necessary to acquire the training dataset. UA-DETRAC is currently not available online, thus it's necessary to contact any researcher or another go-to person from the University of Albany.

It's first necessary to preprocess the annotations into a YOLO-ready format. Then, another preprocessing step needs to be conducted to separate the videos into individual frames. Finally, a sub-sampling method is performed to keep only one frame every ten samples, to make the dataset less equal within itself, reducing overfit probability, and simplifying training.

Then, it's the whole dataset is uploaded to Roboflow to create the dataset ready for deployment. It is available at https://universe.roboflow.com/altice-lzmaq/vehicle-detection-and-tracking-l2epo and it's also strongly recommended for users to start the replicability of our experiment from here.

With the dataset properly organized and ready, it's possible now to begin training. We called several YOLO lightweight detectors and trained those models on variable configurations, to ensure optimizability and avoid crashes (some models are really heavy and it's possible to lose ours of training due to memory overflowing).

We save the best.pt models for each version trained. Then, inference is conducted on the 40 unseen videos of the dataset (it's now again mandatory to perform the same processing steps as before for this new subset) to retreive several performance values, corresponding to valuable metrics such as Precision, Recall, mAP@0.5, mAP@[0.5:0.95], loss values, etc...

Now, with the detection phase already analyzed, it's time to jump into the tracking phase. But before that, it's important to train the ReID module of DeepSORT on the VeRI-776 dataset, again available through a direct message to the Beijing University of Posts and Telecommunications. We basically replicate the backbone structure of the ReID module and retrain it the dataset, generating two versions of the same module: one standard, and another enhanced. This will establish a direct comparison between those approaches.

All YOLO detectors and DeepSORT variants are integrated between each other to perform detection and tracking for the 40 unseen test videos. Reports are generated across 10 confidence thresholds to give insights about performance for all confidences. These reports are then passed to the UA-DETRAC toolkit, which computes both the AP@0.7 and several other tracking metrics, with the latter one being the most useful, for a specific region of interest in the image. From here, tracking is properly analyzed and fit for reporting.

A final experiment can be done in the end to visually insect how the system detects and tracks each vehicle in real-time, while producing a valuable report on the vehicle usage in the end.

Any questions, please address joaomatiasgoncalves321@hotmail.com
