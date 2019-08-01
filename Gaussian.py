import numpy as np
from scipy.stats import multivariate_normal

class Gaussian:
    def __init__(self, mean=np.zeros((3,1)), sigma=np.eye(3)):
        self.mean = np.array(mean)
        self.sigma = np.array(sigma) + np.eye(3)*1e-7

    def compute_probability(self, x):
        return multivariate_normal.pdf(np.array(x),mean=self.mean,cov=self.sigma)

    def update_parameters(self, data):
        self.mean = np.mean(data, axis=0)
        self.sigma = np.cov(data, rowvar=0) + np.eye(3)*1e-7
        
    def set_parameter(self,mean,sigma):
        self.mean = np.array(mean)
        self.sigma = np.array(sigma)
#hello