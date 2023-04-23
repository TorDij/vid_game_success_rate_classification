
# Video Game Success Rate Classification

![Video game console](https://images.unsplash.com/photo-1580327344181-c1163234e5a0?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=867&q=80 "Video Games Sale Based on Rating")
#### For SC1015 Mini Project
NTU School of Computer Science and Engineering
- Group 7
- Lab group - **Z133**

Members
- [Torrey Robert Dijong, Leader](https://github.com/TorDij)
- [Thant Htoo Aung](https://github.com/jack-thant)

Using Jupyter Notebook and Python 3.7

## Problem Statement
Prediction of the success rate of a video game based on the given features of the dataset

## Dataset used
[Video game sales with rating](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)

## Variables Description
| Variables       | Description                                                   |
|:----------------|:--------------------------------------------------------------|
| Name            | Name of the video game                                        |
| Platform        | Game's Platform                                               |
| Year_of_Release | Year of Release                                               |
| Genre           | Game Genre                                                    |
| Publisher       | Name of Publisher                                             |
| NA_Sales        | North American Sales of the video game                        |
| EU_Sales        | European Sales of the video game                              |
| JP_Sales        | Japanese Sales of the video game                              |
| Other_Sales     | Remaining Region Sales of the video game                      |
| Global_Sales    | Global Sales of the video game                                |
| Critic_score    | Aggregate score compiled by Metacritic staff                  |
| Critic_count    | The number of critics used in coming up with the Critic_score |
| User_score      | Score by Metacritic's subscribers                             |
| User_count      | Number of users who gave the user_score                       |
| Developer       | Party responsible for creating the game                       |
| Rating          | The ESRB ratings                                             

### Topic covered in this notebook
1. Data Preparation and cleaning
2. Exploratory Data Analysis
3. ML Models Analysis
    - Logistic Regression
    - Random Forest Classification
    - Ridge Classification
    - SGD Classification
4. Hyperparameter Tuning using GridSearchCV
    - Logistic Regression
    - Random Forest Classification
    - Ridge Classification
    - SGD Classification
5. Model Evaluation

## Introduction
Our group wants to explore the factors to determine the success factor of a video game. For a game to be successful, we defined it to be a game that has either global or regional sales that are higher than the median number of sales in the respective category. As such, we aim to build an accurate prediction model that could potentially predict the success of a game based on the given features in our data set.

### How do we know whether a video game is successful?
For a good definition of the term **Success**, our team decided to determine the success of a game based on the number of sales a game made compared to the rest of the population of games

As such, the game would be successful if any of the following conditions were met:
- The Global_Sales is greater than or equal to the median Global_Sales
- The NA_Sales is greater than or equal to the median NA_Sales
- The EU_Sales is greater than or equal to the median EU_Sales
- The Other_Sales is greater than or equal to the median Other_Sales

Based on the conditions above, a game would be considered successful if it were either globally above average or regionally above median average in sales.

## Data Cleaning

We discovered that we had many missing values for a few of our columns, with over 50% of total values were null for the columns 'User_Count', 'Critic_Score' and 'Critic_Count', as well as 'User_Score' which we discovered that many of the values under 'User_Score' were recorded as 'tbd' (effectively null value).
About 40% of total values were also missing for columns 'Rating' and 'Developer'.

These missing values take up a majority, if not close to a majority of the total data points, which could affect the accuracy of our prediction models as the remaining data available might not be representative of the total data set.

However, based on the source, Kaggle, it was said that this data set only has about 6,900 complete cases.

After some observation, we discovered that by removing rows that had 3 or more missing values for any of the columns, we would be left with much fewer missing values. This is likely because if they were incomplete rows of data, they would likely contain at least 3 missing column values.

Since some columns like Publisher and Developer were missing only a few values, we manually added the values by searching up the details online.

Critic Score and User Score were also missing a few values of around 6%. However, since they are useful values that may be beneficial as predictors of our model, we chose to use mean imputation to fill in the remaining values.

After cleaning up the missing values, we were just left with the removal of outliers.

We also used a Min Max Scaler on the data to ensure that all the values were on the same scale so as to optimise the results for our machine learning models.

## Exploratory Data Analysis
To get a better understanding of our data, we plotted the Year of Release, Genre, Rating, Platform and Publisher against Global Sales to observe any trends in the dataset.

We plotted a boxplot to see the correlation between categorical values like genre, publisher, rating and platform against the regional and global sales.

From our EDA, we concluded that Genre, Publisher and Year of Release were the optimal factors that we can use as predictors in our prediction models.

## ML Models Analysis (Before hyperparameter tuning)
To evaluate our classification models, we will be using precision, accuracy, recall and f1 score. We cannot solely depend on the accuracy metric since our data might be underfit or overfit. We will be using confusion matrix to see the amount of **True Positive** and **True Negative** which our model correctly predict the outcomes.

- **Precision** : Accuracy of positive predictions
- **Recall** : Fraction of positives that were correctly identified
- **F1 score** : What percent of positive predictions were correct?

Below, we will be using classification models such as logistic regression, random Forest Classification, Stochastic Gradient Descent and Ridge Classification.

### 1. Logistic Regression
Logistic Regression is a good model to use as it is good in handling non-linear relationships between predictors and response variable.

Goodness of Fit of Model - Train Dataset:
- Accuracy: 0.67
- Precision: 0.82
- Recall: 0.70
- F1 Score: 0.75

Goodness of Fit of Model - Test Dataset:
- Accuracy: 0.67
- Precision: 0.81
- Recall: 0.71
- F1 Score: 0.75

### 2. Ridge Classifier
The Ridge classifier algorithm works by finding the hyperplane that separates the two classes in the input space. The hyperplane is chosen to minimize the sum of the squared errors between the predicted labels and the true labels, subject to a regularization constraint

Goodness of Fit of Model - Train Dataset:
- Accuracy: 0.67
- Precision: 0.82
- Recall: 0.7
- F1 Score: 0.75

Goodness of Fit of Model - Test Dataset:
- Accuracy: 0.66
- Precision: 0.81
- Recall: 0.71
- F1 Score: 0.75

### 3. Random Forest Classifier
A random forest classifier is a type of ensemble learning algorithm used for classification tasks in machine learning. It is called a "forest" because it consists of multiple decision trees, where each tree is trained on a different subset of the input data, and the final prediction is made by aggregating the predictions of all the trees.

Goodness of Fit of Model - Train Dataset:
- Accuracy: 0.67
- Precision: 0.82
- Recall: 0.7
- F1 Score: 0.75

Goodness of Fit of Model - Test Dataset:
- Accuracy: 0.66
- Precision: 0.81
- Recall: 0.71
- F1 Score: 0.75

### 4. Stochastic Gradient Descent (SGD)
SGDClassifier stands for Stochastic Gradient Descent Classifier. It is a type of linear classification algorithm that is particularly useful for large-scale machine learning problems, where the number of training examples is very large.

Goodness of Fit of Model - Train Dataset:
- Accuracy: 0.26
- Precision: 0.0
- Recall: 0.0
- F1 Score: 0.0

Goodness of Fit of Model - Test Dataset:
- Accuracy: 0.25
- Precision: 0.0
- Recall: 0.0
- F1 Score: 0.0

## Hyperparameter tuning using GridSearchCV
Grid Search finds the optimal hyperparameter values for the other models to improve their performance.

### 1. Logistic Regression (Parameters used)
    - C: the inverse of the regularization strength. A smaller value of C results in stronger regularization, which can reduce overfitting by penalizing large weights in the model. A larger value of C results in weaker regularization, which can lead to better fit to the training data but may also increase the risk of overfitting.
    - penalty: type of regularization (L2 shrinks the weights towards zero without making them exactly zero.)
    - solver: liblinear - the algorithm used to optimise the logistic regression model.

#### Classification report (After GridSearchCV)
Goodness of Fit of Model - Train Dataset
- Accuracy: 	 0.79
- Precision: 	 0.73
- Recall: 	 0.92
- F1 Score: 	 0.81

Goodness of Fit of Model - Test Dataset
- Accuracy: 	 0.75
- Precision: 	 0.80
- Recall: 	 0.89
- F1 Score: 	 0.84

### 2. Random Forest Classifier (Parameters used)
    - n_estimators: The number of decision trees to be used in the random forest
    - max_depth: The maximum depth of each decision tree in the random forest. The deeper the tree, the more complex relationships in the data but risk overfitting.
    - min_samples_split: The minimum number of smaples required to be at a leaf node in a decision tree. Increasing the value will prevent overfitting.
    - bootstrap: a bootstrap parameter that specify whether bootstrap samples should be used when building decision trees. True: each tree is trained on a bootstrap sample of the training data.

#### Classification report (After GridSearchCV)
Goodness of Fit of Model - Train Dataset
- Accuracy: 	 0.91
- Precision: 	 0.90
- Recall: 	 0.92
- F1 Score: 	 0.91

Goodness of Fit of Model - Test Dataset
- Accuracy: 	 0.67
- Precision: 	 0.81
- Recall: 	 0.74
- F1 Score: 	 0.77

## 3.Ridge classification (parameter used)
    - alpha: control the strength of the L2 regularization penalty applied to the coefficents of the model. A large value of alpha results in stronger regularization whcih can results in overfitting.

#### Classification report (After GridSearchCV)
Goodness of Fit of Model - Train Dataset
- Accuracy: 	 0.56
- Precision: 	 0.53
- Recall: 	 0.96
- F1 Score: 	 0.68

Goodness of Fit of Model - Test Dataset
- Accuracy: 	 0.75
- Precision: 	 0.77
- Recall: 	 0.95
- F1 Score: 	 0.85

## 4. SGD Classification (parameters used)
    - alpha: controls the strenght of the L2 (Ridge) regularization penalty applied to the model's coefficients
    - penalty: the type of regularization penalty to be applied
    - max_iter: controls the maximum number of epochs that the model will run during training.
    - tol: tolerance of the stopping criterion based on the loss function

#### Classification report (After GridSearchCV)
Goodness of Fit of Model - Train Dataset
- Accuracy: 	 0.63
- Precision: 	 0.78
- Recall: 	 0.70
- F1 Score: 	 0.74

Goodness of Fit of Model - Test Dataset
- Accuracy: 	 0.62
- Precision: 	 0.78
- Recall: 	 0.67
- F1 Score: 	 0.72

## Model Evaluation
Based on the models above, we can deduce that **logistic regression** is the best model to predict the success rate of our video games. Accuracy, Precision, Recall and F1 score between the train data and test data does not show a lot of difference meaning that the model is neither underfitted nor overfitted. The accuracy is around 75% whereas precision, recall and F1 score are around 85%.

## Acknowledgements
 - [Scikit Learn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
 - [Classification of video game sales for more than 1 million](https://www.kaggle.com/code/ignacioch/predicting-vg-hits-1-million-sales-with-lr-rfc#2.-Prediction-model)
 - [Pandas](https://pandas.pydata.org/)
 - [Seaborn](https://seaborn.pydata.org/)

 ## License
[MIT](https://choosealicense.com/licenses/mit/)


## Badges
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

