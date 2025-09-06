# German Credit Dataset Analysis: Loan Default Prediction

The goal of this project was to proccess and analyze the german credit dataset. With the end result being the generation of several ML models trained to predict the "credibility" or credit worthiness of a potential debtor. 

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Models Implemented](#Machine-Learning-Models-Overview)
- [Results](#results)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Citation](#citation)
- [License](#license)

## Project Overview
**Business Problem:**
For a bank, deciding whether or not to lend an individual money is a critical and extremely important decision. The models developed in this repo can work to increase the chance of a "correct" lending decision. That is, one that minimizes risk of default. 

**Objective:**
Analyze the German credit dataset and develop predictive models that can be used to determine credit worthiness. 

**Approach:**
The approach taken was as follows:

### **First:**
EDA (exploratory data analysis) needs to be performed to get a feel for the data and determine the variables to be used for the predictive models. The EDA performed roughly followed the process below.

**EDA:**
1. **Clean:** Remove any duplicates and assess for missing values.  
2. **Analyze:** Analyze your data and see what variables pertain to your goal (*ANOVA, Chi-Squared, histograms, summaries, boxplots* etc.). Normalization and standardization may be performed.  
3. **Outliers:** Assess for extreme outliers  
4. **Visualize and document:** Report the findings above and proceed to any further analysis.
---
### **Second:**
Selecting the dataset: Performing the (EDA) should have provided enough information to make an informed decision about which variables are relevant and most predictive of credit risk. The selected variables would have been prepared with randomization and a test/training split.

---
### **Third:**
Training predictive models on the previously selected dataset and summarizing the results.
## Dataset Information
**Source:** German Credit Dataset from UCI Machine Learning Repository 

**Description:**
- Number of instances: 1000
- Number of features: 21
- Target variable: "Credibility", this is a binary value where 1 indicates good credit and 0 indicates poor credit.
# Machine Learning Models Overview

## Model 1: Logistic Regression

**Algorithm:** Logistic Regression  
**Purpose:** Baseline model for binary classification  
**Key Features:**

- Implemented using sklearn LogisticRegression model

- Basic linear classifier suitable for establishing performance baseline

- Computationally efficient 
  
  ## Model 2: Random Forest

**Algorithm:** Random Forest  
**Purpose:** Builds multiple decision trees and merges their results to improve accuracy and control over fitting  
**Key Features:**

- Handles both classification and regression tasks
- Robust to over fitting due to averaging over multiple trees
- Provides feature importance rankings
- Works well with mixed data types

## Model 3: Neural Network

**Algorithm:** Deep Neural Network  
**Purpose:** Deep learning method capable of modeling complex non-linear relationships  
**Key Features:**

- 4-layer architecture: 14 → 64 → 32 → 1 neuron layout
- Hidden layers with varying dropout rates for regularization
- Suitable for capturing complex patterns in data
- Requires careful hyperparameter tuning

## Model 4: Support Vector Machine (SVM)

**Algorithm:** Support Vector Machine  
**Purpose:** Supervised learning method for classification that finds the optimal hyperplane to separate data  
**Key Features:**

- Effective in high-dimensional spaces
- Memory efficient (uses subset of training points)
- Generally robust against overfitting in low-noise datasets

## Model 5: XGBoost

**Algorithm:** Extreme Gradient Boosting  
**Purpose:** Gradient boosting framework optimized for speed and performance  
**Key Features:**

- Ensemble method that builds additive decision trees in a sequential manner
- Highly scalable and optimized for speed (parallel processing)
- Built-in regularization to prevent overfitting
- Excellent performance on structured/tabular data

### R Analysis

- Histograms for continuous numerical features (age, credit duration, and credit amount)
- Box plots for continuous numerical features against credibility results
- Basic summary tools

### Python Analysis

- ANOVA for continuous numerical features to confirm relationship with target
- Chi Squared for Nominal and Ordinal categorical variables to find p-value and confirm relationship with target

## Tuning

The goal of this project was not to output a perfectly tuned model so all tuning efforts were relatively tame. \

The NN was manually tuned with hyper parameters.

All other models were tuned with the BayesSearchCV method from the skopt library. The method uses Bayesian search to efficiently apply different tuning parameters through a pipeline. 
This leads to much higher efficiency and intelligent sampling. This combined with the cross validation performed on all parameter combinations selected leads to a comprehensive all-in-one solution that greatly speeds up testing.

All models except for the NN had SMOTETomek applied to them; this was due to the relative skew in the dataset. This was found to improve performance in all metrics. 
Using just SMOTE alone tanked good/bad credit recall, which is exactly the opposite of what is desired. Combining SMOTE with Tomek however seemed to generate the best of both worlds.



# Results

SVM: F1:0.7120147812793642
LrG: F1:0.7073759115477491
XGb: F1:0.7866976845266319
NN: F1: unavailable 
RF: F1:0.7019653387382827

**For all confusion tables and importance graphs please review LINK**

When reviewing model performance, it is important to consider evaluation metrics beyond just accuracy. In imbalanced data sets such as the one used in this credit scoring problem accuracy can be misleading, as a model may perform well simply by predicting the majority class. Therefore, metrics like F1-score, which balances both precision and recall, offer a more reliable view of how well the model performs, especially on the minority class.

Comparing the cross validated F1 scores across top models, we see that the SVM model shows slightly lower overall accuracy than Logistic Regression, while achieving a higher average F1-score, indicating better balance between precision and recall. This suggests that the SVM is more effective at correctly identifying both good and bad credit cases, rather than favoring the dominant class. In a risk sensitive application like credit assessment, this trade-off is often preferable, as failing to detect bad credit cases can lead to significant financial losses.

Taking all factors into account, XGBoost emerges as the strongest overall model. It is tied for the highest accuracy among all models, while also achieving the highest cross-validated F1-score. This indicates that XGBoost not only performs consistently across different data splits, but also maintains a strong balance between class-specific performance. Its robustness, flexibility, and ability to handle imbalanced data make it a highly suitable candidate for deployment in real world credit scoring applications.

### Best Performing Model

XGboost was found to be the best preforming model. Combined with scalability and efficiency it has been deemed to be the recommended model.

### **Feature Importance**

The following were deemed to be the most influential factors when determining a person's credibility/credit worthiness ([expand on this topic](PATH)):

1. **Account balance**
2. **Age** 
3. **Credit history**

---

### **Conclusion**

In conclusion, it has been found that the XGBoost model performs the best when classifying binary credit risk, delivering the highest accuracy and cross-validated F1-score among the tested models. Its ability to effectively handle imbalanced data and capture complex relationships in the dataset makes it a strong candidate for real-world credit scoring applications. Future work should focus on optimization and scale.

---

## Technologies Used

### Prerequisites

- Python 3.13.7
- R (version 4.5.1)
- Git

### Python

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- PyTorch

### R

- tidyverse

### Development Tools

- VS Code

- Git/GitHub

- Rstudio
  
  ## Future Improvements

- [ ] Further tune models for higher accuracy

- [ ] Implement GUI

- [ ] Expand selection of models tested

## Citation

### Dataset

Hofmann, H. (1994). Statlog (German Credit Data) [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.

### This Project

```
[Maxim Faber]. (2025). German Credit Dataset Analysis: Loan Default Prediction. 
GitHub. https://github.com/maximfaber/german-credit-analysis
```

## License

Apache 2.0

---

**Author:** [Maxim Faber]  
**Contact:** [max_F12@protonmail.com]  
**Date:** [2025-09-06]
