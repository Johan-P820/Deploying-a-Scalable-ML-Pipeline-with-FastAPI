# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model Type: RandomForestClassifier

Library: scikit-learn

Key Hyperparameters: min_samples_split=30, default n_estimators

Target Task: Binary classification of income level

Prediction: Whether an individual’s income is >50K or <=50K

Version: v1.0

This model was trained as part of a production ML pipeline using FastAPI.

## Intended Use

Primary Use: Predicting whether an individual earns above or below $50K annually, based on demographic and employment attributes.

Intended Users: Data analysts, ML engineers, and application systems making income-related predictions.

Use Cases:

High-level socioeconomic analysis

Educational or academic modeling exercises

Demonstrations of ML deployment pipelines

## Training Data

Dataset: UCI Census Income Dataset

Features Used:

Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country

Numerical: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week

Label: salary (>50K or <=50K)

Split: 80% Training / 20% Test

Preprocessing:

OneHotEncoder for categorical features

LabelBinarizer for target

No oversampling or undersampling applied

## Evaluation Data

Held-out test set from the same dataset (20% split).

Data was processed using the same encoder and label binarizer used for training.

## Metrics
Precision = 0.78
Recall = 0.61
F1 Score = 0.69

## Ethical Considerations

Model is trained on real demographic data, which may contain historical bias.

Predictions may correlate with protected attributes (sex, race, native country).

Use in decision-making contexts could reinforce inequities.

Labels represent income and are influenced by systemic social disparities.

Mitigation steps recommended:

Perform fairness audits before deployment in any socioeconomically impactful context.

Never use predictions without human judgment and domain context.

## Caveats and Recommendations

Model performance varies across demographic subgroups

Predictions should not be interpreted as causal.

Model should be retrained if:

Demographics shift

Input schema changes

Distribution drifts over time

Recommendation:
If considering real-world deployment, implement continuous monitoring, fairness review, and human oversight.