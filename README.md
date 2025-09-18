# Codsoft3
A simple but powerful machine learning project that classifies iris flowers into species (Setosa, Versicolor, Virginica) using the K-Nearest Neighbors (KNN) algorithm.

This repo contains:

IRIS.csv – Dataset (Iris flower dataset in CSV format).

 iris_model.py – Python script to train & evaluate the KNN classifier.

 Features

 Loads and preprocesses the Iris dataset
 Scales features using StandardScaler
Trains a KNN model (k=5)
Evaluates performance with Accuracy, Classification Report, Confusion Matrix
 Ready-to-run script for quick experimentation

 Dataset Preview (IRIS.csv)
SepalLength	SepalWidth	PetalLength	PetalWidth	Species
5.1	3.5	1.4	0.2	setosa
7.0	3.2	4.7	1.4	versicolor
6.3	3.3	6.0	2.5	virginica
Installation & Usage

1Clone this repository:

git clone https://github.com/your-username/iris-knn-classifier.git
cd iris-knn-classifier


2️ Install dependencies:

pip install -r requirements.txt


3 Run the model:

python iris_model.py

Sample Output
Accuracy: 0.9667

Classification Report:
              precision    recall  f1-score   support
setosa          1.00      1.00      1.00        10
versicolor      0.93      1.00      0.96        14
virginica       1.00      0.92      0.96        16

