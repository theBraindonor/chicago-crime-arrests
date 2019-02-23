# Completed Experiments

Please note that the following experiments were conducted without setting a random_state value.
These models are for exploration of the data only. Final models will be constructed with
repeatability in mind.

The goal for most models with hyper-parameters is to run 24 iterations using BayesSearchCV.

## Decision Tree

Performed on a 10% Sample (613445 records); 24 Iterations each Sampling Strategy

```
No Sampling:
Accuracy: 0.879103
ROC AUC:  0.804535
Log-Loss: 0.311071

Random Under-Sampling:
Accuracy: 0.859209
ROC AUC:  0.821728
Log-Loss: 0.358624

SMOTE Sampling:
Accuracy: 0.874447
ROC AUC:  0.816630
Log-Loss: 0.330552

SMOTEENN Sampling:
Accuracy: 0.850145
ROC AUC:  0.816559
Log-Loss: 0.592685
```

## XGBoost

Performed on a 10% Sample (613445 records); 24 Iterations each Sampling Strategy

```
No Sampling:
Accuracy:
ROC AUC:
Log-Loss:

Random Under-Sampling:
Accuracy:
ROC AUC:
Log-Loss:

SMOTE Sampling:
Accuracy:
ROC AUC:
Log-Loss:

SMOTEENN Sampling:
Accuracy:
ROC AUC:
Log-Loss:
```
