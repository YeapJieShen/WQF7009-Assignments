# Data

The datasets are not pushed to GitHub (`data/*` is gitignored except this file). Download them as described below.

## Stroke Prediction (Assignment 1)

- **Source:** https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
- **File:** `data/healthcare-dataset-stroke-data.csv`
- **Contents:** 5,110 patients, 12 columns: `id`, 10 features (gender, age, hypertension, heart disease, ever married, work type, residence type, average glucose level, BMI, smoking status) and the target `stroke`.
- **Target:** 249 strokes (4.9%); 201 BMI values are missing.

## Brain Tumor Classification MRI (Assignment 2)

- **Source:** https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet (mirror: https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
- **Scripted download (recommended):** run `notebooks/04_mri_eda.ipynb`. It downloads the repository at its only commit (`3ea9f4e`) into `data/brain_mri_github.zip`, extracts it if `data/brain_mri/` is missing, and checks every image against the archive by SHA-256.
- **Manual download:** copy the contents of the repository's `Training` and `Testing` folders into `data/brain_mri/training` and `data/brain_mri/testing`.
- **Manual labels:** `data/brain_mri_manual_labels.csv` (committed) holds the author's visual labels from notebook 04: CT / MRI / Unclear and MRI sequence for the reviewed images. They are judgements by a non-expert, not clinically validated.

Expected layout (3,264 JPEG images):

```
data/brain_mri/
├── training/   glioma_tumor 826, meningioma_tumor 822, no_tumor 395, pituitary_tumor 827
└── testing/    glioma_tumor 100, meningioma_tumor 115, no_tumor 105, pituitary_tumor 74
```
