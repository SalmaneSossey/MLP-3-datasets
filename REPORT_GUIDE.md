# Report Guide - TP2 MLP on 3 Datasets

Use this structure directly for your final report.

## 1. Objective

State the goal: evaluate MLP behavior across three dataset types.
- Regression (Weather WW2)
- Binary classification (Water Potability)
- Multiclass image classification via tabularized pixels (Brain Tumor MRI)

## 2. Experimental Setup

- Environment: Python version, core libraries (PyTorch, scikit-learn, NumPy, pandas)
- Hardware: CPU/GPU
- Reproducibility settings: random seeds, split strategy
- Data source: Kaggle dataset IDs

## 3. Dataset A - Weather WW2 (Regression)

### 3.1 Data Preparation
- Input features selected
- Date/time feature extraction
- Missing-value strategy
- Scaling strategy
- Train/validation/test split proportions

### 3.2 Model and Training
- MLP architecture (layers, activations, dropout)
- Loss function
- Optimizers compared (Adam, AdamW, SGD with momentum)
- Epochs, batch size, learning rate, weight decay, early stopping

### 3.3 Results
- Validation metrics: MAE, RMSE, R2
- Test metrics: MAE, RMSE, R2
- Learning curves (loss)

### 3.4 Discussion
- Which optimizer performed best and why
- Error trends and limitations

## 4. Dataset B - Water Potability (Binary Classification)

### 4.1 Data Preparation
- Feature set and target
- Missing-value handling
- Stratified split
- Scaling
- Class balance inspection

### 4.2 Model and Training
- MLP architecture
- BCEWithLogitsLoss and `pos_weight` rationale
- Hyperparameters and early stopping

### 4.3 Results
- Accuracy, F1-score (validation/test)
- Confusion matrix
- Threshold analysis (if used)

### 4.4 Discussion
- Precision/recall tradeoff
- Impact of class imbalance handling

## 5. Dataset C - Brain Tumor MRI (Multiclass Classification)

### 5.1 Data Preparation
- Image loading and label extraction
- Resize, grayscale conversion, flattening
- Train/validation/test stratified split
- Standardization

### 5.2 Model and Training
- MLP architecture for multiclass output
- Cross-entropy loss
- Optimizer and training setup

### 5.3 Results
- Validation/test accuracy
- Per-class behavior (if available)
- Confusion matrix (recommended)

### 5.4 Discussion
- Why flattening works with MLP but loses spatial structure
- Expected gains from CNN baselines

## 6. Cross-Dataset Comparison

- Compare optimization stability
- Compare metric behavior across task types
- Discuss data representation effects on MLP performance

## 7. Limitations and Improvements

- Data quality constraints
- Hyperparameter search scope
- Potential improvements: better feature engineering, regularization tuning, CNN baseline for image task

## 8. Conclusion

Summarize key findings and practical lessons from using MLPs across heterogeneous datasets.

## 9. Appendix (Optional)

- Full hyperparameter table
- Additional plots
- Reproducibility details
