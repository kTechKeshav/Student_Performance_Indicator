# 🎓 Student Performance Indicator

![GitHub repo size](https://img.shields.io/github/repo-size/kTechKeshav/Student_Performance_Indicator)
[![Render Status](https://img.shields.io/badge/Deployed-Live%20on%20Render-success?style=for-the-badge&logo=render)](https://testforestfires-dyi0.onrender.com/predict_data)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)


A machine learning-based web application built with Flask to **predict a student's performance score** based on multiple academic and behavioral factors like study hours, sleep hours, past scores, etc.

> 🚀 Live Demo: [Click here to try](https://student-performance-indicator-ea1b.onrender.com/predict)

---

## 📸 Preview

![App Screenshot](images\Screenshot.png)

---

## 🧠 Features

- 📊 Predicts a student's performance score using Ridge Regression.
- 📝 Input fields for:
  - Hours Studied
  - Previous Scores
  - Extracurricular Activities (0=No, 1=Yes)
  - Sleep Hours
  - Sample Question Papers Practiced
- 🎨 Stylish, dark-themed UI using HTML & CSS.
- 🔁 Real-time prediction (Flask-powered).

---

## 📦 Tech Stack

| Technology | Description |
|------------|-------------|
| `Python`   | Core language |
| `Flask`    | Backend web framework |
| `HTML/CSS` | Frontend |
| `Sklearn`  | Ridge Regression Model |
| `Render`   | Deployment |

---

## 🔢 How It Works

1. The user fills in academic metrics.
2. Inputs are scaled using `StandardScaler`.
3. Ridge Regression model predicts the performance score.
4. Result is displayed below the form dynamically.

---

## 🚀 Run Locally

Clone the project:

```bash
git clone https://github.com/kTechKeshav/Student_Performance_Indicator.git

cd Student_Performance_Indicator
```

## 📁 Project Structure
```
Student_Performance_Indicator/
├── models/
│   ├── ridge.pkl
│   └── scaler.pkl
├── templates/
│   ├── index.html
│   └── result.html
├── application.py
├── requirements.txt
└── Screenshot.png
```

## 🧠 Model Info
The prediction model was trained using Ridge Regression, which is effective for multicollinearity and prevents overfitting.

## 🙌 Author
Made with ❤️ by Keshav Awasthi

## ⭐ Feedback
If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub. Your support motivates me to keep building cool stuff!