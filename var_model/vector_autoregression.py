#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement the Vector autogression model according to Lutkepohl, 2005. The most
of the code used as reference p.70 ~ p.75, from chapter 3 of the book 'New
Introduction to Multiple Time Series Analysis'.

@author: ucaiado

Created on 09/06/2016
"""
# import libraries
import numpy as np
from numpy.linalg import inv
import pandas as pd

'''
Begin help functions
'''


class FooException(Exception):
    """
    FooException is raised by ...
    """
    pass


class NotFitException(Exception):
    """
    NotFitException is raised by VectorAutoregression to indicate that there is
    no model created
    """
    pass


'''
End help functions
'''


class VectorAutoregression(object):
    '''
    Create a model from a stationary multivariate time series using lagged
    values
    '''
    def __init__(self, df_Y):
        '''
        Initialize a VectorAutoregression object. Save all parameters as
        attributes. The data passed is assumed to be stationary
        :param df_Y: dataframe. a time series from the oldest to newest data
        '''
        self.df_Y = df_Y
        self.na_y = df_Y.values  # (N x K)
        self.b_already_fitted = False
        self.i_p = None  # VAR order
        self.na_A = None  # parameters estimator
        self.na_v = None  # intercep estimator
        self.na_Z = None

    def fit(self, i_p, b_report=False):
        '''
        Fit a VAR model to the data using a i_p lag order. Save the parameters
        found as attributes
        :param i_p: integer. Lag order
        :*param b_report: boolean. print a report of the results
        '''
        # initialize variable
        self.i_p = i_p  # VAR order
        self.na_A = None  # parameters estimator
        self.na_v = None  # intercep estimator
        i_N, i_K = self.df_Y.shape  # number of observations and variables
        i_T = i_N - i_p  # size of the sample
        self.already_fitted = True
        # reshape matrices
        self.na_Z = self._get_z_mat(i_p)  # Z := [Z_0, ..., Z_T] (T x (Kp+1))
        na_Z = self.na_Z.copy()  # ((Kp+1) x T)
        na_ysample = self.na_y[i_p:].T  # Y := (y_1, ..., y_T)  (K x T)
        # measure betas (Lutkepohl, p. 72):
        #    \hat{B} := YZ'(ZZ')^{-1}  (K x (Kp+1))
        # given that B:= (v, A_1, A_2, ..., A_p)  (K x (Kp+1))
        na_ZZTransp_inv = inv(np.dot(na_Z, na_Z.T))
        na_betahat = np.dot(np.dot(na_ysample, na_Z.T), na_ZZTransp_inv)
        # apply vec(\hat{B}) to produce coefficient matrices (K x K)
        # vec is a column stacking operator  (Lutkepohl, p. 70)
        # The first term is the constant. Exclude it from A
        self.na_betahat = na_betahat
        self.na_A = na_betahat[:, 1:].T.reshape((i_p, i_K, i_K))
        self.na_v = na_betahat[:, 0]
        # Transpose the matrices inside A. I shouldnt have to do that
        for idx in xrange(len(self.na_A)):
            self.na_A[idx] = self.na_A[idx].T
        # reshape beta

        # measure the covariance matrix of the error
        # U ~ N(0, \Sigma_u) is the error matrix,  (K x T)
        # \Sigma_u =E[U * U.T] = \frac{1}{T-Kp -1}*U*U.T,  (K x K)
        # is a consistent estimator (Lutkepohl, p. 75)
        na_U = na_ysample - np.dot(na_betahat, na_Z)  # (K x T)
        self.na_U = na_U
        na_Sigma = np.dot(na_U.T, na_U)
        na_Sigma = na_Sigma/(i_T - float(i_K * i_p) - 1.)  # (K x K)
        self.na_Sigma = na_Sigma.T
        # calculate the information criterias

        # print the report
        if b_report:
            pass

    def forecast(self, i_h):
        '''
        Return the forecast based on the model fitted
        :param i_h: integer. Number of steps to forecast ahead
        '''
        if not self.b_already_fitted:
            s_err = u'É necessário fitar um modelo antes de prever'
            raise NotFitException(s_err)
        pass

    def select_order(self, i_p):
        '''
        Return a dataframe with the values of fpe, aic, hq and sc tests
        :param i_p: integer. The maximum number of Lag orders to test
        '''
        pass

    def _get_z_mat(self, i_p):
        '''
        Reshape data to be using the model in the form Z := [Z_0, ..., Z_T]
        (T x Kp), Where T is the sample size and K refers to variables
        :param i_p: integer. The maximum number of Lag orders to test
        '''
        # initiate variables
        na_y = self.na_y
        na_one = np.array([1.])
        # reshape data to be in a column vector of the form
        # Z_t := [[1], [y_t], [y_{t-1}], ... [y_{t - p + 1}]] ((Kp+1) x 1)
        l_Z = []
        for t in xrange(i_p, len(na_y)):
            l_idx = range(t - 1, t - i_p - 1, -1)
            na_Zt = np.hstack([na_one, na_y[l_idx].ravel()])
            l_Z.append(na_Zt.T.copy())
        # Z := [Z_0, ..., Z_T] ((Kp+1) x T)
        na_Z = np.array(l_Z).T

        return na_Z
