#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement the Vector autogression model according to Lutkepohl, 2005. The most
of the codes used as reference p.70 ~ p.75, from chapter 3 of the book 'New
Introduction to Multiple Time Series Analysis'.

@author: ucaiado

Created on 09/06/2016
"""
# import libraries
import numpy as np
from numpy.linalg import inv
import pandas as pd
import scipy.stats as stats

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
        self.b_already_fitted = True
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
        self.na_A = na_betahat[:, 1:].T.reshape((i_p, i_K, i_K))
        self.na_v = na_betahat[:, 0]
        # reshape beta
        self.na_betahat = np.vstack([self.na_v, np.vstack(self.na_A)])
        # Transpose the matrices inside A. I shouldnt have to do that
        for idx in xrange(len(self.na_A)):
            self.na_A[idx] = self.na_A[idx].T
        # measure the covariance matrix of the error
        # U ~ N(0, \Sigma_u) is the error matrix,  (K x T)
        # \Sigma_u =E[U * U.T] = \frac{1}{T-Kp -1}*U*U.T,  (K x K)
        # is a consistent estimator (Lutkepohl, p. 75)
        na_U = na_ysample - np.dot(self.na_betahat.T, na_Z)  # (K x T)
        self.na_U = na_U
        na_Sigma = np.dot(na_U, na_U.T)
        na_Sigma = na_Sigma/(i_T - float(i_K * i_p) - 1.)  # (K x K)
        self.na_Sigma = na_Sigma.T
        # calculate the information criterias

        # print the report
        if b_report:
            pass

    def forecast(self, na_y, i_h, f_alpha=0.05, b_return_interval=True):
        '''
        Return the forecast based on the model fitted and a significance
        interval based on the alpha passed.
        :param na_y: numpy array. The data used in prediction
        :param i_h: integer. h-step predictor. The forecast horizon
        :param f_alpha: float. significance level desired
        '''
        # forecast horizon should be greater than 0
        assert i_h > 0, u'O horizonte de forecast (i_h) deve ser maior que 0'
        # na_y should have the order of var plus the horizon interval - 1
        # s_err = u'Y deve ter {} observacoes'.format(len(self.na_A))
        # assert len(na_y) == len(self.na_A), s_err
        # check if the model is already fitted
        if not self.b_already_fitted:
            s_err = u'E necessario fitar um modelo antes de prever'
            raise NotFitException(s_err)
        # initiate variables
        i_n = len(self.na_A[0])
        na_rtn = np.zeros((1, i_n))
        na_min = np.zeros((1, i_n))
        na_max = np.zeros((1, i_n))
        f_zvalue = stats.norm.ppf(1. - f_alpha / 2.)
        # invert the order to access the newest data first
        l_values = list(na_y)[::-1]
        # apply: E_t[t_{t+1}] = v + A_1 y_t + A_2 y_{t-1} + ... + A_p y_{t-p+2}
        # (Lutkepohl, p. 32 ~ 37)
        for h in xrange(i_h):
            na_y_rtn = self.na_v.copy()  # sum the intercept
            for idx, na_A_i in enumerate(self.na_A):
                na_y_t = l_values[idx].copy()  # t is 0, t-1 is 1 ...
                na_y_rtn += np.dot(na_A_i, na_y_t)  # sum A_t * y
            # append the y_rtn to l_values as it was the first y_t
            l_values = [na_y_rtn.copy()] + l_values
        na_rtn = l_values[0]
        # check if should terminate the function at this point
        if not b_return_interval:
            return na_rtn
        # compute MA coeficient matrix (Lutkepohl, p. 23)
        na_all_phis = self._estimate_ma_phis(i_h)
        # estimate the interval by applying (Lutkepohl, p. 38 ~ 41)
        # there is a different method in p. 97 ~ p. 98. Maybe I am wrong
        na_sigma_y = np.zeros((i_n, i_n))
        # apply: Sigma(h) = Sigma(h-1) + Phi Sigma_u Phi'
        for h in range(i_h):
            na_phi = na_all_phis[h]
            na_var_aux = np.dot(na_phi, np.dot(self.na_Sigma, na_phi.T))
            na_sigma_y += na_var_aux
        self.na_sigma_y = na_sigma_y
        # calculate the interval
        na_sigma = np.diag(self.na_sigma_y) ** .5
        na_min = na_rtn - f_zvalue * na_sigma
        na_max = na_rtn + f_zvalue * na_sigma

        return na_rtn, na_max, na_min

    def _estimate_ma_phis(self, i_h):
        '''
        Estimate the i_h first phis of a MA process
        :param i_h: integer. The number of steps
        '''
        i_order, k, k = self.na_A.shape
        na_phis = np.zeros((i_h+1, k, k))
        # fill the first term with a identity matrix
        # \Sigma_y = I \Sima_u I'
        na_phis[0] = np.eye(k)
        # measure the Phis
        for i in range(1, i_h + 1):
            for j in range(1, i+1):
                if j > i_order:
                    break
                na_phis[i] += np.dot(na_phis[i-j], self.na_A[j-1])

        return na_phis

    def select_order(self, i_max_p):
        '''
        Return a report with the values of FPE, AIC, HQ and SC(BIC) tests
        :param i_max_p: integer. The maximum number of Lag orders to test
        '''
        d_rtn = {}
        df_Y = self.df_Y.copy()
        i_N, i_K = self.df_Y.shape
        for i_ord in xrange(1, i_max_p + 1):
            d_rtn[i_ord] = {}
            i_T = i_N - i_ord
            f_ord = i_ord * 1.
            obj_var = VectorAutoregression(df_Y.copy())
            obj_var.fit(i_ord)
            f_det_Sigma_u = np.linalg.det(obj_var.na_Sigma.copy())
            # measure FPE (Lutkepohl, p. 147)
            f_cst = (i_T + i_K * f_ord + 1.) / (i_T - i_K * f_ord - 1.)**i_K
            d_rtn[i_ord]['FPE'] = f_cst * f_det_Sigma_u
            # measure AIC
            d_rtn[i_ord]['AIC'] = np.log(f_det_Sigma_u)
            d_rtn[i_ord]['AIC'] += (2. * f_ord * i_K**2.)/i_T
            # measure HQ
            d_rtn[i_ord]['HQ'] = np.log(f_det_Sigma_u)
            f_cst = (2. * np.log(np.log(i_T * 1.))) / (i_T) * f_ord * i_K**2
            d_rtn[i_ord]['HQ'] += f_cst
            # measure BIC
            f_cst = (np.log(i_T * 1.)) / (i_T) * f_ord * i_K**2
            d_rtn[i_ord]['SC(BIC)'] = np.log(f_det_Sigma_u)
            d_rtn[i_ord]['SC(BIC)'] += f_cst

        # make the table
        df_rtn = pd.DataFrame(d_rtn)
        df_rtn = df_rtn.round(4).T
        df_rtn.index.name = 'Ordem'

        # check the minium order by criteria
        s_msg = '  Critério {}:  \t\tOrd. {}'
        print 'Ordem com menor valor para cada Critério:'
        for s_col in ['FPE', 'AIC', 'HQ', 'SC(BIC)']:
            print s_msg.format(s_col, np.argmin(df_rtn.ix[:, s_col]))
        print '\n\n\n'
        print df_rtn

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
