import math
import numpy as np
from scipy import optimize

def portfolio_variance(o1, o2, cov):
    """
    Args:
        o1 - standard deviation of portfolio one
        o2 - standard deviation of portfolio two
        cov - covariance of portfolio one and two
    Returns:
        a portfolio variance function that takes allocations of stocks
    """
    def pv(x):
        w1 = x[0]
        w2 = 1 - w1

        print('v:', w1*w1 * o1*o1 + w2*w2 * o2*o2)
        print('cov:', 2 * w1 * w2 * cov)
        v = w1*w1 * o1*o1 + w2*w2 * o2*o2 + 2 * w1 * w2 * cov
        return v
    return pv


o1 = 0.0148179163612915
o2 = 0.0156460042577109
cov = 0.00011389576
x = optimize.minimize(portfolio_variance(o1, o2, cov), [0.55], tol=1e-6)
print(x)

# for i in range(5500, 6000):
#     x = float(i) / 10000.0
#     print(x, '->', portfolio_variance([x]))

def sharpe_ratio(r1, r2, rf, o1, o2, cov):
    """Function to calculate sharpe ratio

    Sharpe ratio is defined by:

        R = Rp - Rf / STDEVp

    Where:

        Rp = Expected portfolio return
        Rf = Risk free rate
        STDEVp = Portfolio standard deviation
    """
    def sr(x):
        w1 = x[0]
        w2 = 1 - w1

        Rp = w1 * r1 + w2 * r2
        STDEVp = math.sqrt(portfolio_variance(o1, o2, cov)(x))
        R = (Rp - rf) / STDEVp
        return R
    return sr

def inverse_sharpe_ratio(r1, r2, rf, o1, o2, cov):
    def isr(x):
        w1 = x[0]
        w2 = 1 - w1

        Rp = w1 * r1 + w2 * r2
        STDEVp = math.sqrt(portfolio_variance(o1, o2, cov)(x))
        R = (Rp - rf) / STDEVp
        return -R
    return isr

R1 = 0.00074693549
R2 = 0.00062554756
Rf = 0
x = optimize.minimize(inverse_sharpe_ratio(R1, R2, Rf, o1, o2, cov), [0.70], tol=1e-8)
print(x)
