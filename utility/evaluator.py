#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import numpy as np
import pandas as pd

from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, log_loss


class Evaluator:
    def __init__(self, logger):
        self.logger = logger

    def evaluate_classifier_result(self, results, test, train=None, test_proba=None, multiclass=False):
        search_results = hasattr(results, 'best_score_')
        if search_results:
            estimator = results.best_estimator_
        else:
            estimator = results

        if search_results:
            self.logger.log('')
            self.logger.log('Best Score:')
            self.logger.log(results.best_score_)
            self.logger.log('')

            self.logger.log('Best Parameters:')
            self.logger.log(results.best_params_)
            self.logger.log('')

        self.logger.log('Classification Report:')
        self.logger.log(classification_report(test.y_actual, test.y_predict))
        self.logger.log('')

        self.logger.log('Confusion Matrix:')
        self.logger.log(pd.DataFrame(confusion_matrix(test.y_actual, test.y_predict)))
        self.logger.log('')

        if train is not None:
            self.logger.log('Accuracy:')
            self.logger.log('   Train: %f' % accuracy_score(train.y_actual, train.y_predict))
            self.logger.log(' Testing: %f\n' % accuracy_score(test.y_actual, test.y_predict))
        else:
            self.logger.log('Accuracy: %f\n' % accuracy_score(test.y_actual, test.y_predict))

        if not multiclass:
            if train is not None:
                self.logger.log('     AUC:')
                self.logger.log('   Train: %f' % roc_auc_score(train.y_actual, train.y_predict))
                self.logger.log(' Testing: %f\n' % roc_auc_score(test.y_actual, test.y_predict))
            else:
                self.logger.log('     AUC: %f\n' % roc_auc_score(test.y_actual, test.y_predict))

        if test_proba is not None:
            self.logger.log('Log-Loss: %f\n' % log_loss(test_proba.y_actual, test_proba.y_predict))

        if 'pca' in estimator.named_steps:
            pca_n_components = estimator.named_steps['pca'].n_components_
            pca_explained_variance = np.sum(estimator.named_steps['pca'].explained_variance_ratio_)
            self.logger.log('PCA:')
            self.logger.log('      N Components: %f' % pca_n_components)
            self.logger.log('Explained Variance: %f' % pca_explained_variance)
            self.logger.log('')
        else:
            features = None
            importance = None

            if 'mapper' in estimator.named_steps:
                features = pd.Series(estimator.named_steps['mapper'].transformed_names_)

            for index, step in estimator.named_steps.items():
                if hasattr(step, 'feature_importances_'):
                    importance = pd.Series(step.feature_importances_)

            if features is not None and importance is not None:
                feature_importance = pd.concat([features, importance], axis=1)
                feature_importance.columns = ['Feature', 'Importance']
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                self.logger.log('Feature Importance:')
                self.logger.log(feature_importance)
                self.logger.log('')

        if search_results:
            self.logger.log('Search Scoring')
            scoring = pd.DataFrame(results.cv_results_)
            scoring = scoring.sort_values('mean_test_score', ascending=False)
            scoring = scoring.filter([
                'mean_test_score', 'std_test_score',
                'mean_train_score', 'std_train_score',
                'mean_fit_time', 'mean_score_time',
                'params'
            ])
            self.logger.log(scoring)
            self.logger.log('')
