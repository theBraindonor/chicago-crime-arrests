# Completed Experiments

Please note that the following experiments were conducted without setting a random_state value.
These models are for exploration of the data only.

The goal for most models with hyper-parameters is to run 12-24 iterations using BayesSearchCV.

# Before Removal of Arrest-Based Crime Reports

## Decision Tree

Performed on a 10% Sample (613445 records); 24 Iterations each Sampling Strategy

```
No Sampling:
Accuracy: 0.885025
ROC AUC:  0.812744
Log-Loss: 0.301356

Random Under-Sampling: ({0.0: 354980, 1.0: 135144}) -> ({0.0: 135144, 1.0: 135144})
Accuracy: 0.859293
ROC AUC:  0.824367
Log-Loss: 0.362333

SMOTE Sampling: ({0.0: 354963, 1.0: 135161}) -> ({0.0: 354963, 1.0: 354963})
Accuracy: 0.877019
ROC AUC:  0.821390
Log-Loss: 0.338944

SMOTEENN Sampling: ({0.0: 355307, 1.0: 134817}) -> ({1.0: 288778, 0.0: 220456})
Accuracy: 0.853898
ROC AUC:  0.821293
Log-Loss: 0.709302
```

## XGBoost

Performed on a 10% Sample (613445 records); 24 Iterations each Sampling Strategy

```
No Sampling:
Accuracy: 0.886633
ROC AUC:  0.815362
Log-Loss: 0.292233

Random Under-Sampling: ({0.0: 354942, 1.0: 135182}) -> ({0.0: 135182, 1.0: 135182})
Accuracy: 0.862663
ROC AUC:  0.829652
Log-Loss: 0.340688

SMOTE Sampling: ({0.0: 355167, 1.0: 134957}) -> ({0.0: 355167, 1.0: 355167})
Accuracy: 0.886233
ROC AUC:  0.816470
Log-Loss: 0.293654

SMOTEENN Sampling: ({0.0: 354948, 1.0: 135176}) -> ({1.0: 287794, 0.0: 219459})
Accuracy: 0.869502
ROC AUC:  0.823661
Log-Loss: 0.376843
```

# After Removal of Arrest-Based Crimes

## Ada Boost

Performed on a 10% Sample (613445 records); 12 Iterations each Sampling Strategy

```
No Sampling:
Accuracy: 0.864172
ROC AUC:  0.661261
Log-Loss: 0.690205

Random Under-Sampling: ({0.0: 354829, 1.0: 77422}) -> ({0.0: 77422, 1.0: 77422})
Accuracy: 0.733424
ROC AUC:  0.747122
Log-Loss: 0.691808

SMOTE Sampling: ({0.0: 354530, 1.0: 77721}) -> ({1.0: 354530, 0.0: 354530})
Accuracy: 0.850097
ROC AUC: 0.676358
Log-Loss: 0.689690

SMOTEENN Sampling: ({0.0: 354715, 1.0: 77536}) -> ({1.0: 317433, 0.0: 223336})
Accuracy: 0.807279
ROC AUC: 0.711530
Log-Loss: 0.690642
```

## Decision Tree

Performed on a 10% Sample (613445 records); 24 Iterations each Sampling Strategy

```
No Sampling:
Accuracy: 0.866161
ROC AUC:  0.675884
Log-Loss: 0.343782

Random Under-Sampling: ({0.0: 354710, 1.0: 77541}) -> ({0.0: 77541, 1.0: 77541})
Accuracy: 0.759835
ROC AUC:  0.750837
Log-Loss: 0.512162

SMOTE Sampling: Original Training Shape is Counter({0.0: 354701, 1.0: 77550}) -> ({0.0: 354701, 1.0: 354701})
Accuracy: 0.855445
ROC AUC:  0.705196
Log-Loss: 0.376332

SMOTEENN Sampling: ({0.0: 354726, 1.0: 77525}) -> ({1.0: 317471, 0.0: 223851})
Accuracy: 0.800246
ROC AUC:  0.737632
Log-Loss: 0.791114
```

## Extra Trees

Performed on a 10% Sample (613445 records); 24 Iterations each Sampling Strategy

```
No Sampling:

Random Under-Sampling:

SMOTE Sampling: 

SMOTEENN Sampling:
```

## Gaussian Naive Bayes

Performed on a 10% Sample

```
No Sampling:

Random Under-Sampling:

SMOTE Sampling: 

SMOTEENN Sampling:
```

## Gradient Boosting

Performed on a 10% Sample (613445 records); 24 Iterations each Sampling Strategy

```
No Sampling:

Random Under-Sampling:

SMOTE Sampling: 

SMOTEENN Sampling:
```

## Neural Network

Performed on a 10% Sample (613445 records)

```
No Sampling:

Random Under-Sampling:

SMOTE Sampling: 

SMOTEENN Sampling:
```

## Random Forest

Performed on a 10% Sample (613445 records); 12 Iterations each Sampling Strategy

```
No Sampling:

Random Under-Sampling:

SMOTE Sampling: 

SMOTEENN Sampling:
```

## Stochastic Gradient Descent

Performed on a 10% Sample (613445 records)

```
No Sampling:

Random Under-Sampling:

SMOTE Sampling: 

SMOTEENN Sampling:
```

## XGBoost

Performed on a 10% Sample (613445 records); 24 Iterations each Sampling Strategy

```
No Sampling:
Accuracy: 0.873111
ROC AUC:  0.684947
Log-Loss: 0.324516

Random Under-Sampling: ({0.0: 354833, 1.0: 77418}) -> ({0.0: 77418, 1.0: 77418})
Accuracy: 0.758918
ROC AUC:  0.758266
Log-Loss: 0.472220

SMOTE Sampling: ({0.0: 354641, 1.0: 77610}) -> ({1.0: 354641, 0.0: 354641})
Accuracy: 0.871816
ROC AUC:  0.688937
Log-Loss: 0.328054

SMOTEENN Sampling: ({0.0: 354523, 1.0: 77728}) -> ({1.0: 317103, 0.0: 222979})
Accuracy: 0.839029
ROC AUC:  0.735930
Log-Loss: 0.390052
```
