# MLP-3-datasets

MLP experiments on three datasets:
- Weather WW2 (regression)
- Water Potability (binary classification)
- Brain Tumor MRI (multiclass classification on flattened images)

The main deliverable is the notebook `notebooks/TP2_MLP_3datasets.ipynb`, organized to support both experimentation and reporting.

## Repository Structure

- `notebooks/TP2_MLP_3datasets.ipynb`: End-to-end TP2 workflow, experiments, plots, and results.
- `src/seed.py`: Reproducibility helpers.
- `src/tabular.py`: Simple tabular dataset utilities.
- `src/image_tabular.py`: Image-to-feature conversion helpers.
- `src/models.py`: MLP model definitions (minimal baseline module).
- `src/train.py`: Lightweight training entrypoint example.
- `src/metrics.py`: Basic metric helpers.
- `requirements.txt`: Python dependencies.
- `REPORT_GUIDE.md`: Report-ready structure and checklist.

## Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional, for Kaggle downloads) place `kaggle.json` at repo root or in `~/.kaggle/`.

## Run the Project

Primary workflow:
- Open `notebooks/TP2_MLP_3datasets.ipynb`
- Run cells in order from top to bottom

Optional module sanity check:

```bash
python src/train.py
```

## Datasets Used

- Weather WW2: `smid80/weatherww2`
- Water Potability: `adityakadiwal/water-potability`
- Brain Tumor MRI: `masoudnickparvar/brain-tumor-mri-dataset`

Downloaded in notebook into:
- `data/weather/`
- `data/potability/`
- `data/brain_tumor/`

## Method Summary

- Weather:
  - Date feature engineering
  - Numeric cleaning/imputation/scaling
  - MLP regression with optimizer comparison

- Potability:
  - Missing-value handling + scaling
  - Stratified split
  - MLP binary classification with class imbalance handling
  - Threshold tuning and confusion-matrix analysis

- Brain Tumor MRI:
  - Convert images to fixed-length vectors (resize + grayscale + flatten)
  - Stratified train/validation/test split
  - MLP multiclass classifier

## Reproducibility Notes

- Use fixed random seeds where possible.
- Keep split logic consistent (`random_state=42`).
- Fit preprocessors on train data only, then transform validation/test.

## Report Preparation

Use `REPORT_GUIDE.md` as the template for:
- Problem framing
- Data processing choices
- Model design and hyperparameters
- Metrics and figures
- Error analysis
- Final conclusions
