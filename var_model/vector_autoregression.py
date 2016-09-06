#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement the Vector autogression model

@author: ucaiado

Created on 09/06/2016
"""
# import libraries
import numpy as np
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
    class __init__(self, df_Y):
        '''
        Initialize a VectorAutoregression object. Save all parameters as
        attributes. The data passed is assumed to be stationary
        :param df_Y: dataframe. a 2-d variables matrix
        '''
        self.b_already_fitted = False
        self.i_p = None  # VAR order
        self.df_A = None  # parameters estimator
        self.df_v = None  # intercep estimator

    class fit(self, i_p):
        '''
        FIT a VAR model to the data using a i_p lag order. Save the parameters
        found as attributes
        :param i_p: integer. Lag order
        '''
        self.i_p = i_p  # VAR order
        self.df_A = None  # parameters estimator
        self.df_v = None  # intercep estimator
        self.already_fitted = True

    class forecast(self, i_h):
        '''
        Return the forecast based on the model fitted
        :param i_h: integer. Number of steps to forecast ahead
        '''
        if not self.b_already_fitted:
            s_err = u'É necessário fitar um modelo antes de prever'
            raise NotFitException(s_err)
        pass

    class select_order(self, i_p):
        '''
        Return a dataframe with the values of fpe, aic, hq and sc tests
        :param i_p: integer. The maximum number of Lag orders to test
        '''
        pass