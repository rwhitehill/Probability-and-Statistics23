#!/usr/bin/env python3

import numpy as np
import math

#############################
# NOTE: CDFs return P(X<=x) #
#############################

###---BINOMIAL---###
def b(x,n,p):
    """
    PMF
    query value:
    - x: number of successes
    parameters:
    - n: sample size
    - p: probability of success
    """
    return math.comb(n,x)*p**x*(1.0-p)**(n-x)

def B(x,n,p):
    """
    CDF
    query value:
    - x: number of successes
    parameters:
    - n: sample size
    - p: probability of success
    """
    temp = 0.0
    i = 0
    while i <= x:
        temp += b(i,n,p)
        i += 1
    return temp

###---Hypergeometric---###
def hg(x,n,M,N):
    """
    PMF
    query value:
    - x: number of successes
    parameters:
    - n: sample size
    - M: maximum number of sucesses in population
    - N: number of objects in population (i.e. maximum sample size)
    """
    return math.comb(M,x)*math.comb(N-M,n-x)/math.comb(N,n)

def HG(x,n,M,N):
    """
    CDF
    query value:
    - x: number of successes
    parameters:
    - n: sample size
    - M: maximum number of sucesses in population
    - N: number of objects in population (i.e. maximum sample size)
    """
    temp = 0.0
    i = 0
    while i <= x:
        temp += hg(i,n,M,N)
        i += 1
    return temp

###---Negative Binomial---###
def nb(x,r,p):
    """
    PMF
    query value:
    - x: number of failures
    parameters:
    - n: sample size
    - r: number of successes
    - p: probability of success
    """
    return math.comb(x+r-1,r-1)*p**r*(1-p)**x

def NB(x,r,p):
    """
    CDF
    query value:
    - x: number of failures
    parameters:
    - n: sample size
    - r: number of successes
    - p: probability of success
    """
    temp = 0.0
    i = 0
    while i <= x:
        temp += nb(i,r,p)
        i += 1
    return temp

###---Poisson---###
def p_(x,mu):
    """
    PMF
    query value:
    - x: number of counts
    parameters:
    - mu: expected number of counts
    """
    return np.exp(-mu)*mu**x/math.factorial(x)

def P_(x,mu):
    """
    CDF
    query value:
    - x: number of counts
    parameters:
    - mu: expected number of counts
    """
    temp = 0.0
    i = 0
    while i <= x:
        temp += p_(i,mu)
        i += 1
    return temp
    
    
    
