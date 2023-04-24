#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Stefan Hiemer
"""
import numpy as np
from numpy import dot, mean, shape
from numpy.linalg import eigh


class PrincipalComponentAnalysis():
    """
    """
    def __init__(self):
        """
        Don't change anything in this function. This  function is not
        necessary, but is just used to tell you which variables you should use.
        We use these variables to check your solution.
        """

        self.eigenvalues, self.eigenvectors = None, None
        self.means = None
        self.covariance_matrix = None

    def train(self, xtrain):
        """
        Calculate the PCA as explained in the script. Please use the variables
        mentioned above. This is techincally not necessary but we use it to
        check your solution and give you hints where something has gone wrong.

        xtrain: np.array of shape (nsamples,ndimensions)
        """

        # calculate class means
        self.means = mean(xtrain, axis=0)

        # set up covariance matrix
        C = xtrain - self.means
        self.covariance_matrix = np.cov(np.transpose(C))

        # calculate eigenvectors of covariance matrix
        self.eigenvalues, self.eigenvectors = eigh(self.covariance_matrix)

        '''self.means = means
        self.covariance_matrix = covariance_matrix
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        '''

        return self.means, self.covariance_matrix, self.eigenvalues, self.eigenvectors

    def transform(self, x):
        """
        Transform data into from the original coordinate system to the
        coordinate system of the principal components.

        x: np.array of shape (nsamples,ndimensions)
        """
        x_transformed = np.transpose(self.eigenvectors).dot(np.transpose(x))
        # each row represents the transformed version of the corresponding row in the input x matrix

        return x_transformed

    def backtransform(self, x_transformed):
        """
        Transform data from the coordinate system of the principal components
        to the original coordinate system.
        """
        x_original = x_transformed.dot(np.transpose(x_transformed)) + self.means

        return x_original


if __name__ == "__main__":
    pass
