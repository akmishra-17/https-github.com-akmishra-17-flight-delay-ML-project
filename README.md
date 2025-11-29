✈️ Flight Delay Prediction using Machine Learning
A Multi-Model Framework for Predicting Commercial Flight Delays

This repository presents a complete end-to-end Machine Learning pipeline designed to analyse airline operational data and predict flight arrival delays.
The project combines supervised learning, unsupervised learning, and feature extraction techniques to build a high-performance prediction system.

🚀 Project Highlights

✔ Full data preprocessing and feature engineering
✔ PCA-based dimensionality reduction
✔ K-Means clustering for pattern discovery
✔ Multiple ML classification models
✔ Extensive evaluation using precision, recall, F1, accuracy & AUC
✔ Over 14 high-quality visualisations generated automatically

🛫 Dataset

The model uses a publicly available Airline On-Time Performance dataset (USA, 2023).
It contains real flight-level metrics including:

Flight timings

Departure & arrival delays

Aircraft taxi times

Origin & destination airports

Carrier information

Target variable:

Arrival Delay ≥ 15 minutes (binary classification)

🧠 Machine Learning Models Implemented
Supervised Models

Logistic Regression

Decision Tree Classifier

Support Vector Machine (SVM)

Random Forest Classifier

K-Nearest Neighbours (KNN)

Unsupervised Model

K-Means Clustering (on PCA-reduced features)

Feature Reduction

Principal Component Analysis (PCA)

Data Processing

StandardScaler

Label Encoding

Missing value handling

Correlation analysis

📊 Generated Visualisations

The script automatically saves all plots, including:

Arrival Delay Distribution

Class Balance

Correlation Heatmap

PCA Scatter Plot

K-Means Cluster Plot

Confusion Matrices for all models

ROC Curves

Feature Importance (Random Forest)

These are saved inside the images/ folder.

🏆 Results Summary

The model comparison shows exceptionally high performance across all classifiers, with Random Forest and SVM providing the strongest results with near-perfect metrics.

📂 Repository Structure
📦 flight-delay-ML-project
 ┣ 📂 images                   → All visualisations
 ┣ 📜 assessment.py            → Main ML pipeline script
 ┣ 📜 README.md                → Project documentation
 ┗ 📜 dataset.csv              → Airline performance dataset

👩‍💻 Author

Aakanksha Mishra
Data Science Student | Machine Learning Enthusiast
Passionate about predictive modelling, analytical storytelling, and building real-world AI applications.

📜 License

This project is open for educational and personal use.
