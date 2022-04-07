---
layout: post
title: "About the Project"
author: "Kelvin Murillo, Shad Fernandez"
categories: about-the-project
tags: [documentation,sample]
image: serImage.jpg
---
# Speech Emotion Recognition Using Deep Learning
## Introduction

The recognition of emotion in human speech is a widely researched topic, as there are many different applications 
that would directly benefit from this technology such as real time customer service as well as providing support for potential 
deaf people. With the ability to accurately decipher someone's emotion based on **how** a sentence is spoken can vavstly 
aid those who are deaf by providing additional social cues that may not be existent due to their condition. A real time 
classification of emotion throughout a conversation would also aid customer service of companies and provide further customer 
satisfaction by classifying the emotion of customers in real time.

## Datasets
The datasets used in the creation of the models we created was an open source dataset collected from kaggle which can be
viewed [here](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en). This dataset contains the following
popular speech emotion datasets all in one: Crema, Ravdess, Savee, and Tess. This dataset contains audio files with labeled
emotions in the filename. The labeled emotions vary from the 4 datasets, but as will be described in the methodology, only 
a certain number of emotions were kept in the final models outputted due to many different factors such as data constraints and
dimensionality reduction in the modeling process. in total, there are 12,142 audio files all in english and saying a multitude
of different sentences with different tones and emotions.



## Methodology
Before modeling our data, we first needed to perform the ETL process as well as clean our data. The ETL process
was straight forward, as all of our data was in a downloaded directory containing sub folders with all of the audio files
within each dataset. After downloading the dataset, we set out to create a pandas dataframe containing the path to the dataframe
along with the features extracted from the audio file, and lastly the labeled emotion taken from the name of the audio file.
In order to extract features from each audio file, we used the audio features that could be easily extracted from the 
[librosa](https://librosa.org/doc/main/feature.html) package. For the traditional machine learning models, we simply 
extracted the mean of each feature, as the feature extraction from librosa returns an array of features where each entry 
represents a specific frame in the audio file. Some of the features extracted include the mean values of Mel-frequency cepstral coefficients
(MFCCs), spectral centroid, and spectral bandwidth among others. 
**TALK ABOUT FEATURES FOR THE NEURAL NET HERE AND HOW IT IS DIFFERENT THAN THE MEANS USED IN THE TRADITIONAL MODELS**
These features were then used to train and evaluate our traditional machine learning models as well as our neural nets.



## Traditional Machine Learning Models
For the traditional machine learning models, we attempted to solve a multi-class classification problem around 4 different 
emotions: happy, sad, angry, and neutral. The models tested include random forests, k nearest neighbors, and decision trees.
The features that are considered most important by the random forest model include melspectograms (especially around 
the 13th - 14th frame) and mfcc values. This is expected, because as seen in past research, MFCC values and melspectogram
values were also deemed important features in these studies as well. In order to evaluate the traditional machine learning
models, we filtered our original dataset into these 4 emotions and then performed an 80/20 train test split where the 
testing data was not touched until the evaluation phase of the pipeline. After training, testing, and evaluating our models,
we found that the random forest model performed best out of the traditional machine learning models, as seen in the results
tab.

## Neural Networks






