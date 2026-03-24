# Stroke Risk Prediction using Machine Learning

## Project Overview
This project focuses on developing a machine learning classification model to predict whether a patient is at risk of a stroke. Stroke is a critical medical condition where blood supply to the brain is interrupted, and early prediction is vital because prevention is significantly more effective than treatment.

## Dataset Description
The data for this project was sourced from Kaggle.
•	Observations: 5,110 patient records.
•	Target Variable: stroke (1 = Patient has had a stroke, 0 = Patient has not had a stroke).
•	Key Features: Gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status.

## Methodology
The project follows a standard data science pipeline:
•	Data Cleaning: Removed unnecessary columns (like id) and handled missing values in the bmi column using median imputation.
•	Feature Encoding: Categorical variables (gender, marriage status, etc.) were converted into numerical format using LabelEncoder.
•	Class Imbalance: Since stroke cases represent less than 5% of the total data, the SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the dataset.

## Algorithms Tested:
- Logistic Regression
-	Decision Tree
-	Random Forest
-	K-Nearest Neighbors (KNN)
- XGBoost

## Key Findings and Results
- Best Performing Model: Logistic Regression outperformed more complex models in terms of Recall.
- Conclusion: In a healthcare context, maximizing Recall is critical to minimize "false negatives" (missing actual stroke cases). The success of Logistic Regression suggests the dataset has relatively simple decision boundaries.

## Required Libraries
To run the notebook, you will need the following Python libraries installed:
- pandas and numpy (Data manipulation)
- matplotlib and seaborn (Visualization)
- scikit-learn (Machine learning and preprocessing)
- imblearn (SMOTE for balancing data)
- xgboost (Advanced modeling)
•	xgboost (Advanced modeling)

