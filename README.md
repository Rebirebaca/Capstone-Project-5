# Singapore Resale Flat Prices Predicting

## Introduction:
This project aims to construct a machine learning model and implement it as a user-friendly online application in order to provide accurate predictions about the resale values of apartments in Singapore.
This prediction model will be based on past transactions involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the worth of a flat after it has been previously resold.

## Skills take away From This Project:
Data Wrangling,
EDA,
Feature Engineering,
Model Building, 
Model Deployment

## Domain: Real Estate

## Key Technologies and Skills:
* Python
* Numpy
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn
* Pickle
* Streamlit
* Render

## Project Workflow:
Data Collection and Preprocessing
Feature Engineering
Model Selection and Training
Model Evaluation
Streamlit Web Application
Deployment on Render
Testing and Validation  

## Workflow overview:

### Data Preprocessing:

#### Data Understanding: 
Before diving into modeling, it's crucial to gain a deep understanding of your dataset. Start by identifying the types of variables within it, distinguishing between continuous and 
categorical variables, and examining their distributions. In our dataset.

### Feature Engineering:

#### Encoding and Data Type Conversion: 
To prepare categorical features for modeling, we employ LabelEncoder encoding. This technique transforms categorical values into numerical representations based on their intrinsic nature
and their relationship with the target variable. Additionally, it's essential to convert data types to ensure they match the requirements of our modeling process.

#### Skewness - Feature Scaling: 
Skewness is a common challenge in datasets. Identifying skewness in the data is essential, and appropriate data transformations must be applied to mitigate it. One widely-used method is the log transformation,
which is particularly effective in addressing high skewness in continuous variables. This transformation helps achieve a more balanced and normally-distributed dataset, which is often a prerequisite for many
machine learning algorithms.

#### Outliers Handling: 
Outliers can significantly impact model performance. We tackle outliers in our data by using the Interquartile Range (IQR) method. This method involves identifying data pointsthat fall outside the IQR boundaries 
and then converting them to values that are more in line with the rest of the data. This step aids in producing a more robust and accurate model.

### Exploratory Data Analysis (EDA) 

#### Skewness Visualization:
To enhance data distribution uniformity, we visualize and correct skewness in continuous variables using Seaborn's Histplot and boxplot. By applying the Log Transformation method,we achieve improved balance and 
normal distribution, while ensuring data integrity.

Outlier Visualization: We identify and rectify outliers by leveraging Seaborn's Boxplot. This straightforward visualization aids in pinpointing outlier-rich features. Our chosen remedy is the Interquartile
Range (IQR) method, which brings outlier data points into alignment with the rest of the dataset, bolstering its resilience.

### Model Selection and Training:

#### Algorithm Selection:
Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests).After thorough evaluation, Random Forest Regressor, demonstrate commendable testing 
accuracy. Upon checking for any overfitting issues in both training and testing, both models exhibit strong performance without overfitting concerns. I choose the Random Forest Regressor for its ability to
strike a balance between interpretability and accuracy, ensuring robust performance on unseen data.

### Model Evaluation:
#### Hyperparameter Tuning with GridSearchCV and Cross-Validation: 
Evaluate model performance using metrics like RMSE, MAE, R².To fine-tune our model and mitigate overfitting, we employ GridSearchCV with cross-validation for hyperparameter tuning. This function allows us to 
systematically explore multiple parameter values and return the optimal set of parameters. {'max_depth': 20, 'max_features': ='log2', 'min_samples_leaf': 2, 'min_samples_split': 5}

### Streamlit Web Application:
The objective is to develop a Streamlit webpage that enables users to input values for each column and get the expected resale_price value for the flats in Singapore.

### Deployment on Render:
Deploy the Streamlit application on the Render platform to make it accessible to users over the internet. Render is a cloud-based platform that makes it easy to deploy and manage web application.

#### Link: https://capstone-project-5-2.onrender.com

### Testing and Validation:
Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.You can test the application by inputting different flat details and
verifying that the predicted resale prices are reasonable.




