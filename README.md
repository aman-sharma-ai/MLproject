# 🎓 End-to-End ML: Student Performance Prediction

Predicting a student's **math score** from demographic and study-preparation data, comparing 9 regression models. The repo also includes a real-time **driver drowsiness detector** built with computer vision.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

---

## 📌 Problem

How much do gender, ethnicity group, parental education, lunch type and test-preparation course affect a student's math score? Can we predict it from those factors plus reading and writing scores?

## 📂 What's inside

| Path | Description |
|---|---|
| [`notebook/1 . EDA STUDENT PERFORMANCE .ipynb`](notebook/) | Data checks, distributions and how each factor relates to scores |
| [`notebook/2. MODEL TRAINING.ipynb`](notebook/) | Preprocessing, training and evaluating 9 regression models |
| [`src/`](src/) | Modular pipeline: logging and custom exceptions done; ingestion, transformation and training components in progress |
| [`Drowsiness_Detection.py`](Drowsiness_Detection.py) | Real-time drowsiness and yawn detection from a webcam |

## 📈 Results

Test-set R² for every model compared:

| Rank | Model | R² |
|---:|---|---:|
| 🥇 | Ridge | **0.881** |
| 🥈 | Linear Regression | 0.880 |
| 🥉 | CatBoost | 0.852 |
| 4 | AdaBoost | 0.852 |
| 5 | Random Forest | 0.849 |
| 6 | XGBoost | 0.828 |
| 7 | Lasso | 0.825 |
| 8 | K-Nearest Neighbors | 0.784 |
| 9 | Decision Tree | 0.735 |

**Takeaway:** the relationship is mostly linear, so simple regularised linear models beat the tree ensembles. Linear Regression reaches **RMSE ≈ 5.4** and **MAE ≈ 4.2** marks on the test set.

## 😴 Drowsiness detection

Uses **dlib's 68-point facial landmarks** to compute the **eye aspect ratio** and **mouth aspect ratio** every frame. If the eyes stay closed (EAR < 0.25) or the driver yawns (MAR > 0.75) for **20 consecutive frames**, an alarm sounds.

Before running it, download [`shape_predictor_68_face_landmarks.dat`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) into a `models/` folder and add an alarm sound named `music.wav`.

## ▶️ Run it locally

```bash
git clone https://github.com/AmanDataverse/MLproject.git
cd MLproject
pip install -r requirements.txt
jupyter notebook            # open the notebooks in /notebook

# drowsiness detector (extra dependencies)
pip install opencv-python dlib imutils pygame scipy
python Drowsiness_Detection.py
```

## 🗺️ Roadmap

- [ ] Finish the `src/` pipeline (ingestion → transformation → training → prediction)
- [ ] Add a Flask or Streamlit app for live predictions
- [ ] Deploy and add a demo link here
