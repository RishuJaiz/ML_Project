# Music Genre Classification using Machine Learning Techniques

## Project Overview

This project is centered on the development of a robust music genre classification system using the GTZAN dataset, which is one of the most utilized datasets in the field of Music Information Retrieval (MIR). The project leverages advanced audio signal processing techniques and machine learning algorithms to accurately predict the genre of music tracks.

The project also involves the development and deployment of a machine learning model designed to perform predictive analysis on GTZAN data. Using Python, the project encapsulates the entire workflow of a predictive modeling task including data preprocessing, feature engineering, model training, model evaluation, and finally, making predictions.

The primary goal of this project is to provide a robust solution that can be utilized to predict outcomes based on historical data. This is achieved through a carefully engineered pipeline that ensures data cleanliness, optimal feature selection, and effective model training strategies.

## Key Objectives

1. **Audio Processing**: Using librosa, a Python library, to handle audio files and extract useful features such as Mel-frequency cepstral coefficients (MFCCs) and various spectral features which are crucial for understanding music content.

2. **Feature Engineering**: Transforming raw audio data into a format suitable for machine learning models, focusing on extracting a comprehensive set of features that capture the unique aspects of different music genres.

3. **Model Development**: Training and testing multiple machine learning models including Support Vector Machines (SVM), Decision Trees, K-Nearest Neighbors (KNN), Naive Bayes, Multi-layer Perceptron (MLP), CatBoost, XGBoost, and AdaBoost. This approach allows us to evaluate and select the best performer based on accuracy and other relevant metrics.

4. **Evaluation**: Assessing the performance of each model to identify the most effective algorithm for classifying music genres. This includes comparing their predictive accuracy, handling of imbalanced data, and their ability to generalize on unseen data.

The objective of this project is not only to achieve high accuracy in genre classification but also to explore the effectiveness of different feature sets and models, providing a comprehensive understanding of the task at hand. This project serves as a valuable resource for anyone looking to delve into the field of music genre classification or expand their knowledge on the application of machine learning in audio analysis.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- Librosa, Pandas, NumPy, Scikit-Learn, TensorFlow, CatBoost, XGBoost
- Jupyter Notebook or compatible IDE

## Dataset

The GTZAN dataset can be downloaded from Kaggle. The audio (.wav) files are contained under the folder 'genres_original' inside the folder 'data'. The folder 'data' contains spectrogram images for the various audio files of different genres in the folder 'images_original'.

We have also extracted the MFCCs, LPC, and various spectral features which are in `windowing_mfcc_dataframe.csv`, `sec03_LPC_extracted_features_final_df.csv`, and `spectral_dataframe.csv` respectively.

## Running the Project

1. Open the Jupyter Notebook:
    ```bash
    jupyter notebook ML_MODEL2.ipynb
    ```

2. Execute the cells sequentially to load the data, perform feature extraction, train the models, and evaluate their performance.

## Output

The project outputs the accuracy of the various models tested, providing insights into which model performs best for music genre classification based on the GTZAN dataset. Plots and metrics are displayed within the notebook for detailed analysis.
