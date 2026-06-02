# Auto-ML-Model-Builder
 A AutoML workflow system that enables non-technical users to design and execute end-to-end machine learning pipelines with automated model selection and optimization.
 
## Overview

The **Block-Based AutoML Workflow System for Business Automation** is a web-based platform designed to simplify machine learning workflows for users with little or no programming experience.

The system currently focuses on three core functionalities:

- 📂 Dataset Upload
- 📊 Automated Data Analysis
- 🤖 Machine Learning Model Recommendation

By automating the initial stages of the machine learning process, the platform helps users quickly understand their data and identify suitable machine learning models.

---

## Features

### 📂 Dataset Upload
- Upload datasets in CSV format.
- Automatic validation of uploaded files.
- Dataset preview before analysis.

### 📊 Automated Data Analysis
The system automatically performs exploratory data analysis (EDA), including:

- Dataset dimensions
- Data types
- Missing value detection
- Statistical summaries
- Feature distribution analysis
- Data quality insights

### 🤖 Model Recommendation Engine
Based on dataset characteristics, the system recommends suitable machine learning algorithms.

#### Classification Examples
- Logistic Regression
- Decision Tree
- Random Forest

#### Regression Examples
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Recommendation criteria include:
- Target variable type
- Feature composition
- Dataset structure

---

## System Workflow

```text
User
 │
 ▼
Dataset Upload
 │
 ▼
Data Validation
 │
 ▼
Automated Data Analysis
 │
 ▼
Model Recommendation Engine
 │
 ▼
Recommended Models
```

---

## Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask

### Machine Learning & Data Processing
- Pandas
- NumPy
- Scikit-learn

### Data Visualization
- Matplotlib
- Seaborn

---

## Current Implementation

### ✅ Completed Features
- Dataset upload
- Data validation
- Automated exploratory data analysis
- Statistical reporting
- Machine learning model recommendation

### 🚀 Future Enhancements
- Drag-and-drop block workflow builder
- Automated model training
- Hyperparameter tuning
- Workflow execution engine
- Model comparison dashboard
- Auto-generated reports
- End-to-end AutoML pipeline generation

---

## Business Value

This platform helps users:

- Reduce manual effort in data analysis
- Understand datasets quickly
- Identify suitable machine learning models
- Accelerate the machine learning workflow
- Enable AI adoption for non-technical users

---

## Future Scope

The long-term vision of this project is to evolve into a complete **Block-Based AutoML Platform** where users can visually design, customize, and execute machine learning pipelines without writing code.

---
