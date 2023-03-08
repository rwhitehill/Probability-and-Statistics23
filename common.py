#!/usr/bin/env python3

import numpy as np
import math
import scipy.optimize as sciop
import scipy.integrate as scint
from scipy.special import gamma

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


###---Normal---###
class NORMAL():
    
    def __init__(self,mu,sig,z_max=10.0):
        """
        parameters:
        - mu: expected value of rv
        - sig: standard deviation of rv
        - z_max: sets limits for integration (z-score)
        """
        self.mu  = mu
        self.sig = sig
        
        self.min   = mu - z_max*sig
        self.max   = mu + z_max*sig
        
        self.exp_val = mu
        self.var     = sig**2.0
        self.std     = sig
        
    def z(x):
        return (x-self.mu)/self.sig
        
    def pdf(self,x):
        return np.exp(-(x-self.mu)**2.0/2.0/self.sig**2.0)/np.sqrt(2.0*np.pi*self.sig**2.0)
    
    def cdf(self,x):
        return scint.quad(self.pdf,self.min,min(x,self.max))[0]
    
    def percentile(self,p):
        func = lambda x: self.cdf(x) - p
        return sciop.root_scalar(func,bracket=[self.min,self.max]).root


###---Exponential---###
class EXP():
    
    def __init__(self,lam):
        """
        parameters:
        - lam: inversely related to mean and standard deviation
        * essentially the continuous version of the poisson distribution
        """
        self.lam = lam
        
        self.min = 0
        self.max = np.inf
        
        self.exp_val = 1.0/lam
        self.var     = self.exp_val**2.0
        self.std     = np.sqrt(self.var)
        
    def pdf(self,x):
        if x >= self.min:
            return self.lam*np.exp(-self.lam*x)
        else:
            return 0

    def cdf(self,x):
        if x >= self.min:
            return 1.0 - np.exp(-self.lam*x)
        else:
            return 0
        
    def percentile(self,p):
        return -np.log(1.0-p)/self.lam
            

###---Gamma---###
class GAMMA():
    
    def __init__(self,alfa,beta):
        self.alfa = alfa
        self.beta = beta
        
        self.exp_val = alfa*beta
        self.var     = alfa*beta**2.0
        self.std     = np.sqrt(self.var)
        
        self.min = 0
        self.max = self.exp_val + 20.0*self.std #np.inf
        
    def pdf(self,x):
        if x >= self.min:
            return x**(self.alfa-1.0)*np.exp(-x/self.beta)/gamma(self.alfa)/self.beta**self.alfa
        else:
            return 0
    
    def cdf(self,x):
        return scint.quad(self.pdf,self.min,min(x,self.max))[0]
        
    def percentile(self,p):
        func = lambda x: self.cdf(x) - p
        return sciop.root_scalar(func,bracket=[self.min,self.max]).root
    
###---Weibull---###
class WEIBULL():
    
    def __init__(self,alfa,beta,gam=0.0):
        self.alfa = alfa
        self.beta = beta
        self.gam  = gam
        
        self.exp_val = beta*gamma(1.0+1.0/alfa) + gam
        self.var     = beta**2.0*(gamma(1.0+2.0/alfa)-gamma(1.0+1.0/alfa))
        self.std     = np.sqrt(self.var)
        
        self.min = gam
        self.max = self.exp_val + 20.0*self.std #np.inf
        
    def pdf(self,x):
        if x >= self.min:
            return self.alfa*(x-self.gam)**(self.alfa-1.0)*np.exp(-((x-self.gam)/self.beta)**self.alfa)/self.beta**self.alfa
        else:
            return 0
    
    def cdf(self,x):
        if x < self.min:
            return 0
        else:
            return 1.0 - np.exp(-((x-self.gam)/self.beta)**self.alfa)
        
    def percentile(self,p):
        func = lambda x: self.cdf(x) - p
        return sciop.root_scalar(func,bracket=[self.min,self.max]).root
      
###---Lognormal---###
class LOGNORMAL():
    
    def __init__(self,mu,sig,z_max=10.0):
        self.mu  = mu
        self.sig = sig
        
        self.exp_val = np.exp(mu+sig**2.0/2.0) 
        self.var     = np.exp(2.0*mu+sig**2.0)*(np.exp(sig**2.0)-1.0)
        self.std     = np.sqrt(self.var)
            
        self.min = 1e-10
        self.max = mu + z_max*self.std
        
    def pdf(self,x):
        if x < self.min:
            return 0
        else:
            return np.exp(-(np.log(x) - self.mu)**2.0/2.0/self.sig**2.0)
    
    def cdf(self,x):
        Z = NORMAL(0.0,1.0)
        return Z.cdf((np.log(x) - self.mu)/self.sig)
    
    def percentile(self,p):
        func = lambda x: self.cdf(x) - p
        return sciop.root_scalar(func,bracket=[self.min,self.max]).root
    
    
###---Beta---###
class BETA():
    
    def __init__(self,alfa,beta,A,B):
        self.alfa = alfa
        self.beta = beta
        self.A    = A
        self.B    = B
        
        self.exp_val = A + (B-A)*alfa/(alfa+beta)
        self.var     = (B-A)**2.0*alfa*beta/(alfa+beta)**2.0/(alfa+beta+1.0)
        self.std     = np.sqrt(self.var)
        
        self.min = A
        self.max = B
    
    def pdf(self,x):
        L = self.B - self.A
        return 1.0/L*gamma(self.alfa+self.beta)/gamma(self.alfa)/gamma(self.beta)*((x-self.A)/L)**(self.alfa-1.0)*((self.B-x)/L)**(self.beta-1.0)
    
    def cdf(self,x):
        if x >= self.A or x <= self.B:
            return scint.quad(self.pdf,self.min,min(x,self.max))[0]
        else:
            return 0
    
    def percentile(self,p):
        func = lambda x: self.cdf(x) - p
        return sciop.root_scalar(func,bracket=[self.min,self.max]).root
        
    
        

            