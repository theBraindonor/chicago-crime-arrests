#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2019, John Hoff"
__license__ = "Creative Commons Attribution-ShareAlike 4.0 International License"
__version__ = "1.0"

import pandas as pd

from collections import Counter

from skopt import BayesSearchCV

from sklearn.model_selection import train_test_split

from utility import batch_predict, batch_predict_proba, EvaluationFrame, Evaluator, Logger, use_project_path


class Runner:
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

    def run_classification_experiment(
            self,
            sample=None,
            random_state=None,
            test_size=0.20,
            multiclass=False,
            record_predict_proba=False,
            sampling=None):
        use_project_path()

        logger = Logger('%s.txt' % self.name)

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

        if self.hyper_parameters is not None:
            self.estimator.set_params(**self.hyper_parameters.params)

        logger.time_log('Training Model...')
        self.estimator.fit(x_train, y_train)
        logger.time_log('Training Complete.\n')

        logger.time_log('Testing Training Partition...')
        y_train_predict = batch_predict(self.estimator, x_train)
        logger.time_log('Testing Complete.\n')

        train_evaluation_frame = EvaluationFrame(y_train, y_train_predict)

        logger.time_log('Testing Holdout Partition...')
        y_test_predict = batch_predict(self.estimator, x_test)
        logger.time_log('Testing Complete.\n')

        test_evaluation_frame = EvaluationFrame(y_test, y_test_predict)
        test_evaluation_frame.save('%s_predict.p' % self.name)

        test_proba_evaluation_frame = None
        if record_predict_proba:
            logger.time_log('Testing Holdout Partition (probability)...')
            y_test_predict_proba = batch_predict_proba(self.estimator, x_test)
            test_proba_evaluation_frame = EvaluationFrame(y_test, y_test_predict_proba)
            test_proba_evaluation_frame.save('%s_predict_proba.p' % self.name)
            logger.time_log('Testing Complete.\n')

        evaluator = Evaluator(logger)
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
