#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    A common training and evaluation runner to allow for easy and consistent model creation and evalutation
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import pandas as pd

from collections import Counter

from skopt import BayesSearchCV

from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle

from utility import batch_predict, batch_predict_proba, EvaluationFrame, Evaluator, Logger, use_project_path, batch_fit_classifier


def crossfold_classifier(estimator, transformer, x_train, y_train, train_index, test_index,
                         record_predict_proba, verbose, fit_increment, warm_start, max_iters, random_state):
    """
    This method allows for training to be done using the joblib parallelism in scikit learn.  Overall a hacky
    method to allow for incremental training.  Really needs to be refactored into a cleaner form.
    """
    if hasattr(x_train, 'iloc'):
        x_fold_train, x_fold_test = x_train.iloc[train_index], x_train.iloc[test_index]
    else:
        x_fold_train, x_fold_test = x_train[train_index], x_train[test_index]

    if hasattr(y_train, 'iloc'):
        y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]
    else:
        y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

    if fit_increment is not None:
        if max_iters is not None:
            for iter in range(max_iters):
                x_fold_train, y_fold_train = shuffle(x_fold_train, y_fold_train, random_state=random_state)
                batch_fit_classifier(estimator, x_fold_train, y_fold_train, transformer=transformer, increment=fit_increment, verbose=verbose)
        else:
            batch_fit_classifier(estimator, x_fold_train, y_fold_train, transformer=transformer, increment=fit_increment, verbose=verbose)
    else:
        if transformer is not None:
            x_fold_train = transformer.transform(x_fold_train)
        estimator.fit(x_fold_train, y_fold_train)

    y_fold_test_predict = batch_predict(estimator, x_fold_test, transformer=transformer, verbose=False)
    fold_predict_frame = EvaluationFrame(y_fold_test, y_fold_test_predict)

    fold_predict_proba_frame = None
    if record_predict_proba:
        y_fold_test_predict_proba = batch_predict_proba(estimator, x_fold_test, transformer=transformer, verbose=False)
        fold_predict_proba_frame = EvaluationFrame(y_fold_test, y_fold_test_predict_proba)

    return Evaluator.evaluate_classifier_fold(fold_predict_frame, fold_predict_proba_frame)


class Runner:
    """
    The runner supports bare estimator fitting and searvh-based fitting.  By default it will make use of the a
    BayesianSearchCV to perform hyperparameter tuning.  Ensures everything is cleanly logged, evaluated, and pickled.
    """
    def __init__(
            self,
            name,
            df,
            target,
            estimator,
            hyper_parameters=None):
        self.name = name
        self.df = df
        self.target = target
        self.estimator = estimator
        self.hyper_parameters = hyper_parameters
        self.trained_estimator = None

    def run_classification_experiment(
            self,
            sample=None,
            random_state=None,
            test_size=0.20,
            multiclass=False,
            record_predict_proba=False,
            sampling=None,
            cv=5,
            verbose=True,
            transformer=None,
            fit_increment=None,
            warm_start=False,
            max_iters=None,
            n_jobs=-1):
        use_project_path()

        logger = Logger('%s.txt' % self.name)
        evaluator = Evaluator(logger)

        data_frame = self.df

        if sample is not None:
            data_frame = data_frame.sample(n=sample, random_state=random_state)

        x_train, x_test, y_train, y_test = train_test_split(data_frame, data_frame[self.target], test_size=test_size)

        if transformer is not None:
            logger.time_log('Fitting Transformer...')
            transformer.fit(x_train)
            logger.time_log('Transformer Fit Complete.\n')

        if sampling is not None:
            logger.time_log('Starting Data Re-Sampling...')
            logger.log('Original Training Shape is %s' % Counter(y_train))
            x_new, y_new = sampling.fit_resample(x_train, y_train)
            logger.log('Balanced Training Shape is %s' % Counter(y_new))
            if hasattr(x_train, 'columns'):
                x_new = pd.DataFrame(x_new, columns=x_train.columns)
            x_train, y_train = x_new, y_new
            logger.time_log('Re-Sampling Complete.\n')
            logger.time_log('Shuffling Re-Sampled Data.\n')
            x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
            logger.time_log('Shuffling Complete.\n')

        if self.hyper_parameters is not None:
            self.estimator.set_params(**self.hyper_parameters.params)

        if cv is not None:
            kfold = StratifiedKFold(n_splits=cv, random_state=random_state)
            logger.time_log('Cross Validating Model...')
            fold_scores = Parallel(n_jobs=n_jobs, verbose=3)(
                delayed(crossfold_classifier)(
                    clone(self.estimator),
                    transformer,
                    x_train, y_train,
                    train_index, test_index,
                    record_predict_proba, verbose,
                    fit_increment, warm_start, max_iters, random_state
                )
                for train_index, test_index in kfold.split(x_train, y_train)
            )
            logger.time_log('Cross Validation Complete.\n')

        logger.time_log('Training Model...')
        if fit_increment is not None:
            if max_iters is not None:
                for iter in range(max_iters):
                    x_iter_train, y_iter_train = shuffle(x_train, y_train, random_state=random_state)
                    batch_fit_classifier(self.estimator, x_iter_train, y_iter_train, transformer=transformer, increment=fit_increment, verbose=verbose)
            else:
                batch_fit_classifier(self.estimator, x_train, y_train, transformer=transformer, increment=fit_increment, verbose=verbose)
        else:
            if transformer is not None:
                x_train_transformed = transformer.transform(x_train)
                self.estimator.fit(x_train_transformed, y_train)
            else:
                self.estimator.fit(x_train, y_train)
        logger.time_log('Training Complete.\n')

        logger.time_log('Testing Training Partition...')
        y_train_predict = batch_predict(self.estimator, x_train, transformer=transformer, verbose=verbose)
        logger.time_log('Testing Complete.\n')

        train_evaluation_frame = EvaluationFrame(y_train, y_train_predict)

        logger.time_log('Testing Holdout Partition...')
        y_test_predict = batch_predict(self.estimator, x_test, transformer=transformer, verbose=verbose)
        logger.time_log('Testing Complete.\n')

        test_evaluation_frame = EvaluationFrame(y_test, y_test_predict)
        test_evaluation_frame.save('%s_predict.p' % self.name)

        test_proba_evaluation_frame = None
        if record_predict_proba:
            logger.time_log('Testing Holdout Partition (probability)...')
            y_test_predict_proba = batch_predict_proba(self.estimator, x_test, transformer=transformer, verbose=verbose)
            test_proba_evaluation_frame = EvaluationFrame(y_test, y_test_predict_proba)
            test_proba_evaluation_frame.save('%s_predict_proba.p' % self.name)
            logger.time_log('Testing Complete.\n')

        if cv is not None:
            evaluator.evaluate_fold_scores(fold_scores)

        evaluator.evaluate_classifier_result(
            self.estimator,
            test_evaluation_frame,
            train=train_evaluation_frame,
            test_proba=test_proba_evaluation_frame,
            multiclass=multiclass
        )

        logger.close()

        if self.hyper_parameters is not None:
            self.hyper_parameters.save('%s_params.p' % self.name)

        self.trained_estimator = self.estimator

    def run_classification_search_experiment(
            self,
            scoring,
            sample=None,
            random_state=None,
            test_size=0.20,
            n_jobs=-1,
            n_iter=2,
            cv=5,
            verbose=3,
            multiclass=False,
            record_predict_proba=False,
            sampling=None):
        use_project_path()

        logger = Logger('%s.txt' % self.name)

        search = BayesSearchCV(
            self.estimator,
            self.hyper_parameters.search_space,
            n_jobs=n_jobs,
            n_iter=n_iter,
            cv=cv,
            verbose=verbose,
            scoring=scoring,
            return_train_score=True
        )

        data_frame = self.df

        if sample is not None:
            data_frame = data_frame.sample(n=sample, random_state=random_state)

        x_train, x_test, y_train, y_test = train_test_split(data_frame, data_frame[self.target], test_size=test_size)

        if sampling is not None:
            logger.time_log('Starting Data Re-Sampling...')
            logger.log('Original Training Shape is %s' % Counter(y_train))
            x_new, y_new = sampling.fit_resample(x_train, y_train)
            logger.log('Balanced Training Shape is %s' % Counter(y_new))
            if hasattr(x_train, 'columns'):
                x_new = pd.DataFrame(x_new, columns=x_train.columns)
            x_train, y_train = x_new, y_new
            logger.time_log('Re-Sampling Complete.\n')
            logger.time_log('Shuffling Re-Sampled Data.\n')
            x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
            logger.time_log('Shuffling Complete.\n')

        logger.time_log('Starting HyperParameter Search...')
        results = search.fit(x_train, y_train)
        logger.time_log('Search Complete.\n')

        logger.time_log('Testing Training Partition...')
        y_train_predict = batch_predict(results.best_estimator_, x_train)
        logger.time_log('Testing Complete.\n')

        train_evaluation_frame = EvaluationFrame(y_train, y_train_predict)

        logger.time_log('Testing Holdout Partition...')
        y_test_predict = batch_predict(results.best_estimator_, x_test)
        logger.time_log('Testing Complete.\n')

        test_evaluation_frame = EvaluationFrame(y_test, y_test_predict)
        test_evaluation_frame.save('%s_predict.p' % self.name)

        test_proba_evaluation_frame = None
        if record_predict_proba:
            logger.time_log('Testing Holdout Partition (probability)...')
            y_test_predict_proba = batch_predict_proba(results.best_estimator_, x_test)
            test_proba_evaluation_frame = EvaluationFrame(y_test, y_test_predict_proba)
            test_proba_evaluation_frame.save('%s_predict_proba.p' % self.name)
            logger.time_log('Testing Complete.\n')

        evaluator = Evaluator(logger)
        evaluator.evaluate_classifier_result(
            results,
            test_evaluation_frame,
            train=train_evaluation_frame,
            test_proba=test_proba_evaluation_frame,
            multiclass=multiclass
        )

        logger.close()

        self.hyper_parameters.params = results.best_params_
        self.hyper_parameters.save('%s_params.p' % self.name)

        self.trained_estimator = results.best_estimator_
