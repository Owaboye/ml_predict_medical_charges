# Medical Insurance Charges Prediction — End-to-End ML Project

## Introduction
This project aims to predict the medical insurance charges for customers based on demographic and health data.  
We'll explore the data, preprocess it, build predictive models, and evaluate their performance — mimicking a real-world data science project.

## Business Problem
Accurate estimation of insurance charges helps insurance companies optimize premiums and assess risk profiles efficiently.
Our goal is to predict `charges` using features like `age`, `bmi`, `smoker`, and `region`.

## Data Description
This dataset contains the following columns:
- `age`: Age of primary beneficiary
- `sex`: Gender
- `bmi`: Body Mass Index
- `children`: Number of children
- `smoker`: Smoking status
- `region`: Residential region
- `charges`: Medical insurance charges (target)

## Load the Data
- Import required libraries (pandas, numpy, matplotlib, seaborn).
- Load the CSV file into a DataFrame. [medical-charges]([URL](https://github.com/Owaboye/ml_predict_medical_charges/blob/main/medical-charges.csv))
```
df = pd.read_csv('medical-charges.csv')
```

### Data Inspection: •	Check the shape, column names, data types.
First, let's look at the data shape and types to understand its structure.
```
df.head()
df.info()
print(df.shape)
print(df.columns)
```

### Univariate Analysis
Here we visualize the distribution of numeric variables like `age`, `bmi`, and `charges`.
#### Why?
This helps us to understand feature distributions, ranges, and potential outliers.

