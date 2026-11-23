# Student Performance Prediction: An End-to-End Machine Learning Project

This project predicts a student's **mathematics score** from demographic and test-preparation data and compares the performance of nine regression models. The repository also includes a real-time **driver drowsiness detection** system built with computer vision.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

---

## Problem Statement

To what extent do gender, ethnicity group, parental level of education, lunch type and completion of a test-preparation course influence a student's mathematics score? Can that score be predicted reliably from these factors together with reading and writing scores?

## Repository Contents

| Path | Description |
|---|---|
| [`notebook/1 . EDA STUDENT PERFORMANCE .ipynb`](notebook/) | Data validation, distribution analysis and the relationship between each factor and student scores |
| [`notebook/2. MODEL TRAINING.ipynb`](notebook/) | Preprocessing, training and evaluation of nine regression models |
| [`src/`](src/) | Modular training pipeline. Logging and custom exception handling are complete; the ingestion, transformation and training components are under development. |
| [`Drowsiness_Detection.py`](Drowsiness_Detection.py) | Real-time drowsiness and yawn detection using a webcam |

## Results

Test-set R² for each model evaluated:

| Rank | Model | R² |
|---:|---|---:|
| 1 | Ridge | **0.881** |
| 2 | Linear Regression | 0.880 |
| 3 | CatBoost | 0.852 |
| 4 | AdaBoost | 0.852 |
| 5 | Random Forest | 0.849 |
| 6 | XGBoost | 0.828 |
| 7 | Lasso | 0.825 |
| 8 | K-Nearest Neighbors | 0.784 |
| 9 | Decision Tree | 0.735 |

**Conclusion:** The relationship between the features and the target is largely linear, so regularized linear models outperform the tree-based ensembles. Linear Regression achieves an **RMSE of approximately 5.4** and an **MAE of approximately 4.2** marks on the test set.

## Drowsiness Detection

The system uses **dlib's 68-point facial landmark model** to calculate the **eye aspect ratio (EAR)** and **mouth aspect ratio (MAR)** for each video frame. An alarm is triggered when the eyes remain closed (EAR below 0.25) or the driver yawns (MAR above 0.75) for **20 consecutive frames**.

Before running the script, download [`shape_predictor_68_face_landmarks.dat`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) into a `models/` directory and provide an alarm audio file named `music.wav`.

## Running Locally

```bash
git clone https://github.com/aman-sharma-ai/MLproject.git
cd MLproject
pip install -r requirements.txt
jupyter notebook            # open the notebooks in /notebook

# Drowsiness detection (additional dependencies)
pip install opencv-python dlib imutils pygame scipy
python Drowsiness_Detection.py
```

## Roadmap

- [ ] Complete the `src/` pipeline (ingestion, transformation, training and prediction)
- [ ] Add a Flask or Streamlit application for live predictions
- [ ] Deploy the application and publish a demonstration link
