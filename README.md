# Breast Cancer Classification using Machine Learning

## Overview
This project implements a machine learning model to classify breast cancer tumors as malignant or benign using the Breast Cancer Wisconsin Dataset. The classification is performed using various machine learning models, including Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines (SVM).

## Dataset
The dataset used for this project is the **Breast Cancer Wisconsin Dataset**, which contains features computed from digitized images of fine needle aspirate (FNA) biopsies of breast masses.

- **Source**: UCI Machine Learning Repository
- **Features**: 30 numerical attributes describing characteristics of cell nuclei
- **Target Variable**: Malignant (1) or Benign (0)

## Dependencies
To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib seaborn scikit
```
```

## Implementation Steps
1. **Data Preprocessing**
   - Handle missing values (if any)
   - Normalize and standardize data
   - Split data into training and testing sets (80%-20%)
   
2. **Model Training**
   - Train multiple classifiers (Logistic Regression, SVM, Random Forest, Decision Tree)
   - Use GridSearchCV for hyperparameter tuning

3. **Model Evaluation**
   - Evaluate models using accuracy, precision, recall, and F1-score
   - Use confusion matrix for performance analysis

## Usage
To train and evaluate the model, run the following command:

```bash
python src/train.py
```

To visualize results:
```bash
python src/evaluate.py
```


## Future Enhancements
- Implement deep learning models (e.g., Neural Networks)
- Explore additional feature selection techniques
- Deploy model as a web API

