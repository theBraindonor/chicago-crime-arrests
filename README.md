# Chicago Crime Arrests

_A Machine Learning Project for Crime Arrests in Chicago, Illinois_

This project was created for DSC 540 Advanced Machine Learning as part of the Master's in Data Science program at
DePaul University in Chicago during the winter 2019 quarter.

For additional information, please refer to the included `Project_Presentation.pdf` and `Project_Paper.pdf`
in the repository root. (forthcoming)

### Author

John Hoff <john.hoff@braindonor.net>

### License

Creative Commons Attribution-ShareAlike 4.0 International License

## Environment Setup

This project requires Python 3.6+ to be installed.  It is highly recommended that a virtual environment be created.

On Windows:
```
python -m virtualenv venv
venv\scripts\activate
pip install -r requirements.txt
```

On OSX and Linux:

```
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Source

The data for this project was taken from the Chicago Data Portal:
https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2

To begin replicating the results of this analysis, exported data should initially be saved in:
`data_scratch/Crimes_-_2001_to_present.csv`.

## Pre-Processing

Generate dictionaries for the category variables.

```
python -m data.categories
```

Pre-Process the data, scrubbing unknowns and converting all categories to numeric values.

```
python -m data.preprocess
```

Review the category aggregations on the full dataset, showing the arrest-based crimes.

```
python -m model.exploration.aggregation_full
```

Remove arrest-based crimes and repeat the category aggregations.

```
python -m data.remove_arrest_based_crime
python -m model.exploration.aggregation
```

Sample the full and cleaned data sets.  The sample data will be used for initial model generation.

```
python -m data.sample
python -m data.sample_cleaned
```

## Initial Model Generation With Arrest-Based Crimes Included

The following models were created using 10% of the data after initial preprocessing.  This
set of data includes arrest-based crimes.  These are crimes where reports are only filed _after_
the arrest has been made.

```
python -m model.experiment.decision_tree_full_model
python -m model.experiment.xgboost_full_model
```

## Initial Model Scoring with Arrest-Based Crimes Included

Scoring calculations of the initial models including arrest-based crimes:

```
python -m model.experiment.full_model_scoring_search
```

## Initial Model Generation

The following models were creating using 10% of the data after full preprocessing.  Please note that some
of these models will take hours to run.

```
python -m model.experiment.ada_boost_model
python -m model.experiment.decision_tree_model
python -m model.experiment.extra_trees_model
python -m model.experiment.gaussian_naive_bayes_model
python -m model.experiment.gradient_boosting_model
python -m model.experiment.neural_network_model
python -m model.experiment.random_forest_model
python -m model.experiment.sgd_huber_loss_model
python -m model.experiment.sgd_log_loss_model
python -m model.experiment.xgboost_model
```

## Initial Model Scoring

Scoring calculations of the initial models are performed by the following:

```
python -m model.experiment.model_scoring_search
```

## Final Model Generation

Final models are created using 100% of the data after pre-processing.  Please note that these models will take
hours to run.

```
python -m model.neural_network_model
python -m model.sgd_huber_loss_model
python -m model.xgboost_model
```

## Final Model Scoring

Final scoring calculations are performed by the following:

```
python -m model.model_scoring_search
```

## Visualizations

ROC AUC and Confusion Matrices:

```
python -m jupyter notebook visualization/roc_curves.ipynb
python -m jupyter notebook visualization/confusion_matrix.ipynb
```

Map Visualization of XGBoost Model:

```
python -m jupyter notebook visualization/xgboost_maps.ipynb
```

Map Visualization of Neural Network Model:

```
python -m jupyter notebook visualization/neural_network_maps.ipynb
```