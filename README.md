# Fake-news-detector
A machine learning-based web application built to detect fake news using natural language processing (NLP) and classification algorithms. This project is still under development, and there are several aspects that need further improvement. The model is trained on the Fake News Detection Dataset available on Kaggle.

Technologies Used

Python

Scikit-learn

Pandas

Streamlit

Joblib

Regex (re)

Overview

This project uses Logistic Regression combined with TF-IDF vectorization to build a fake news detection model. The model is trained on a dataset containing both real and fake news articles. The app allows users to input news articles and check if they are real or fake.

Files

train.py – Script to train the fake news detection model.

app.py – Streamlit web app for deploying the trained model.

model.pkl – Trained model saved as a pickle file.

vectorizer.pkl – Saved vectorizer for text preprocessing and feature extraction.

Installation

To run this project, you'll need to set up the environment with the required dependencies.

1. Clone the repository:
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection

2. Install required libraries:
pip install pandas scikit-learn joblib streamlit

3. Train the model:

Make sure you have the Fake News Detection Dataset from Kaggle,and place the dataset in your project folder. After placing the dataset, update the paths in train.py accordingly.

To train the model, run:

python train.py


This will train the logistic regression model, evaluate its performance, and save both the model and vectorizer as model.pkl and vectorizer.pkl respectively.

Usage
1. Run the Streamlit app:

After training the model, run the Streamlit web app with the following command:

streamlit run app.py

2. Interact with the web app:

Enter a news article in the provided text box.

Click "Predict" to get a prediction of whether the news is real or fake.

The result will be displayed as either "Fake News" or "Real News."

Features

Text Preprocessing: News text is preprocessed by converting to lowercase, removing non-word characters, and normalizing spaces.

Model Training: Logistic Regression model is trained on TF-IDF features extracted from n-grams.

Prediction: The trained model predicts whether the entered news is real or fake based on the provided text.

Development Status

This project is under development. Several features are still in progress, including improving the model's accuracy, expanding the dataset, and adding more advanced techniques such as deep learning models. Contributions to the project are welcome.

Contributing

Feel free to fork the repository and make improvements! If you'd like to contribute, please submit a pull request with a detailed description of the changes.

License

This project is licensed under the MIT License - see the LICENSE
 file for details.

Acknowledgements

Scikit-learn: For machine learning tools and algorithms.

Streamlit: For building the web application easily.

Pandas: For data manipulation and processing.

Kaggle: For the Fake News Detection Dataset used in training the model.
