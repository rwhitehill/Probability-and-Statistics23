#!/usr/bin/env python3

import numpy as np
import math

#############################
# NOTE: CDFs return P(X<=x) #
#############################



#==================================================#


##########################
# DISCRETE DISTRIBUTIONS #
##########################


###---BINOMIAL---###
class BINOM():
    
    def __init__(self,n,p):
        """
        query value:
        - x: number of successes
        parameters:
        - n: sample size
        - p: probability of success
        """
            
        self.n = n
        self.p = p
        self.q = 1.0-p
        
        self.min = 0
        self.max = n
        
        self.exp_val = n*p
        self.var     = n*p*self.q
        self.std     = np.sqrt(self.var)
        
    def pmf(self,x):
        return math.comb(self.n,x)*self.p**x*self.q**(self.n-x)


###---Hypergeometric---###
class HYPER_GEOM():
    
    def __init__(self,n,M,N):
        """
        query value:
        - x: number of successes in sample
        parameters:
        - n: sample size
        - M: maximum number of sucesses in population
        - N: number of objects in population (i.e. maximum sample size)
        """

        self.n = n
        self.M = M
        self.N = N
        
        self.min = max(0,n-N+M)
        self.max = min(n,M)
        
        self.exp_val = n*M/N
        self.var     = (N-n)/(N-1)*self.exp_val*(1.0-M/N)
        self.std     = np.sqrt(self.var)
        
    def pmf(self,x):
        return math.comb(self.M,x)*math.comb(self.N-self.M,self.n-x)/math.comb(self.N,self.n)

    
###---Negative Binomial---###
class NEG_BINOM():
    
    def __init__(self,r,p):
        """
        query value:
        - x: number of failures before rth success
        parameters:
        - n: sample size
        - r: number of successes
        - p: probability of success
        """
        
        self.r = r
        self.p = p
        
        self.min = 0
        self.max = np.inf
        
        self.exp_val = r*(1.0-p)/p
        self.var     = self.exp_val/p
        self.std     = np.sqrt(self.var)
        
        
    def pmf(self,x):
        return math.comb(x+self.r-1,self.r-1)*self.p**self.r*(1.0-self.p)**x
    

###---Poisson---###
class POISSON():
    
    def __init__(self,mu):
        """
        query value:
        - x: number of counts
        parameters:
        - mu: expected number of counts
        """
        
        self.mu = mu
        
        self.min = 0
        self.max = np.inf
        
        self.exp_val = mu
        self.var     = mu
        self.std     = np.sqrt(self.var)
        
    def pmf(self,x):
        return np.exp(-self.mu)*self.mu**x/math.factorial(x)

def discrete_cdf(rv,x):
    temp = 0.0
    i = rv.min
    while i <= min(x,rv.max):
        temp += rv.pmf(i)
        i += 1
    return temp

    
#==================================================#


############################
# CONTINUOUS DISTRIBUTIONS #
############################


