# Vertebral Column Classification

Binary classification project to predict vertebral column abnormalities (Normal vs Abnormal) using K-Nearest Neighbors (KNN) algorithm with various distance metrics.

## Overview

This project implements a KNN classifier from scratch to classify vertebral column conditions based on biomechanical features. The implementation includes:

- **Multiple distance metrics**: Euclidean, Manhattan, Chebyshev, Minkowski, and Mahalanobis distances
- **Weighted KNN**: Inverse distance weighting for improved predictions
- **Comprehensive evaluation**: Error analysis, learning curves, and performance metrics
- **Exploratory data analysis**: Pairwise scatterplots and boxplots for feature visualization

## Dataset

The Vertebral Column dataset contains 310 instances with 6 biomechanical features:
- Pelvic incidence
- Pelvic tilt
- Lumbar lordosis angle
- Sacral slope
- Pelvic radius
- Grade of spondylolisthesis

**Target variable**: Binary classification (Normal=0, Abnormal=1)

**Data split**: 
- Training set: 210 instances (70 Normal, 140 Abnormal)
- Test set: 100 instances (30 Normal, 70 Abnormal)

## Features

- Custom KNN implementation from scratch
- Multiple distance metrics comparison
- Weighted voting mechanism
- Learning curve analysis
- Comprehensive performance metrics (TPR, TNR, Precision, F1-Score)
- Exploratory data analysis with visualizations

## Results

### Best Performance
- **Best k (Euclidean)**: 16
- **Test Error**: 0.1000 (10%)
- **True Positive Rate**: 1.0000
- **Precision**: 0.8750
- **F1-Score**: 0.9333

### Distance Metrics Comparison
| Metric | Best k* | Test Error |
|--------|---------|------------|
| Manhattan | 6 | 0.1100 |
| Minkowski (log10(p)=0.6) | 6 | 0.1000 |
| Chebyshev | 16 | 0.1000 |
| Mahalanobis | 1 | 0.1700 |

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vertebral-column-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the notebooks directory:
```bash
cd notebooks
```

2. Open and run the Jupyter notebook:
```bash
jupyter notebook vertebral_column_classification.ipynb
```

The notebook will:
- Load and preprocess the data
- Perform exploratory data analysis
- Train and evaluate KNN models with different distance metrics
- Generate visualizations and performance metrics

## Project Structure

```
vertebral-column-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── vertebral_column_data/
│       ├── column_2C.dat
│       ├── column_2C_weka.arff
│       ├── column_3C.dat
│       └── column_3C_weka.arff
├── notebooks/
│   └── vertebral_column_classification.ipynb
└── results/
    ├── pairplot.png
    ├── boxplots.png
    ├── error_curve_euclidean.png
    └── learning_curve.png
```

## Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualizations
- **Jupyter Notebook**: Interactive development and analysis

## Key Implementation Details

### KNN Algorithm
- Custom implementation without using scikit-learn
- Supports multiple distance metrics
- Includes weighted voting option
- Efficient neighbor search with sorting

### Distance Metrics
- **Euclidean**: Standard L2 norm distance
- **Manhattan**: L1 norm distance (city block)
- **Chebyshev**: Maximum coordinate difference
- **Minkowski**: Generalized distance with parameter p
- **Mahalanobis**: Distance accounting for covariance structure

## Performance Metrics

The project evaluates models using:
- **Error Rate**: Overall misclassification rate
- **True Positive Rate (TPR/Recall)**: Sensitivity
- **True Negative Rate (TNR/Specificity)**: Specificity
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Acknowledgments

- Dataset: UCI Machine Learning Repository - Vertebral Column Dataset
- This project demonstrates fundamental machine learning concepts and custom algorithm implementation
