import math
import numpy as np
from scipy import optimize

def portfolio_variance(o, cov):
    """
    Args:
        o - standard deviation vector of portfolio
        cov - covariance matrix of portfolio
    Returns:
        a portfolio variance function that takes allocations of stocks
    """
    def pv(x):
        """
        Args:
            x - allocations of stocks, needs to sum to 1
        """
        sq_x = np.square(x)
        sq_o = np.square(o)
        v = sq_x.dot(sq_o)
        for i in range(len(o)):
            w1 = x[i]
            for j in range(i+1, len(o)):
                w2 = x[j]
                v += 2 * w1 * w2 * cov[i, j]
        return v
    return pv


def sharpe_ratio(r, rf, o, cov):
    """Function to calculate sharpe ratio

    Sharpe ratio is defined by:

        R = Rp - Rf / STDEVp

    Where:

        Rp = Expected portfolio return
        Rf = Risk free rate
        STDEVp = Portfolio standard deviation

    Args:
        r - expected return of individual stocks vector
        rf - risk free rate
        o - standard deviation of portfolio vector
        cov - covariance matrix of portfolio
    """
    def sr(x):
        """
        Args:
            x - stock allocations vector
        """
        Rp = x.dot(r)
        STDEVp = math.sqrt(portfolio_variance(o, cov)(x))
        R = (Rp - rf) / STDEVp
        return R
    return sr

def inverse_sharpe_ratio(r, rf, o, cov):
    def isr(x):
        Rp = x.dot(r)
        STDEVp = math.sqrt(portfolio_variance(o, cov)(x))
        R = (Rp - rf) / STDEVp
        return -R
    return isr


def msft_vs_wfc():
    o = np.array([0.0148179163612915, 0.0156460042577109])
    cov = np.array([
        [0, 0.00011389576],
        [0.00011389576, 0],
    ])
    constraints = (
        {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1,
        }
    )
    w = np.array([0.5, 0.5])
    x = optimize.minimize(portfolio_variance(o, cov), w, tol=1e-15, constraints=constraints)
    print("Minimum risk portfolio:")
    print(x)
    print()

    # for i in range(5500, 6000):
    #     x = float(i) / 10000.0
    #     print(x, '->', portfolio_variance(o, cov)([x, 1-x]))

    r = np.array([0.00074693549, 0.00062554756])
    rf = 0
    x = optimize.minimize(inverse_sharpe_ratio(r, rf, o, cov), w, tol=1e-8, constraints=constraints)
    print("Tangent portfolio (best return / risk ratio):")
    print(x)
    mean = r.dot(x.x)
    stdev = math.sqrt(portfolio_variance(o, cov)(x.x))
    sharpe = sharpe_ratio(r, rf, o, cov)(x.x)
    print('Mean:', mean * 100, '%')
    print('Stdev:', stdev * 100, '%')
    print('Sharpe ratio:', sharpe)

def all_stocks(long_only=False):
    o = np.array([
        0.01651612375,
        0.02494045222,
        0.01676328139,
        0.01373353017,
        0.01599824533,
        0.01481791636,
        0.03421804815,
        0.02483268137,
        0.01564600426,
        0.01197595924,
    ])
    cov = np.array([
    	[0, 0.000140203631, 0.000086396286, 0.000079833364, 0.000103346079, 0.000091197689, 0.000123845154, 0.000126246273, 0.000099530540, 0.000067677521],
    	[0, 0, 0.000135793623, 0.000130760584, 0.000170832048, 0.000126159095, 0.000213293471, 0.000221281568, 0.000154767539, 0.000098042449],
    	[0, 0, 0, 0.000101608813, 0.000086662554, 0.000095832082, 0.000140269147, 0.000156523798, 0.000131737077, 0.000143214807],
        [0, 0, 0, 0, 0.000091763337, 0.000093516874, 0.000131567610, 0.000139403830, 0.000131328467, 0.000088054013],
    	[0, 0, 0, 0, 0, 0.000107775720, 0.000134732040, 0.000118864479, 0.000107988527, 0.000074676417],
        [0, 0, 0, 0, 0, 0, 0.000114217126, 0.000133822184, 0.000113895764, 0.000081578374],
        [0, 0, 0, 0, 0, 0, 0, 0.000222685764, 0.000136963603, 0.000097438987],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.000172849564, 0.000118691480],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000110245131],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    constraints = (
        {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1,
        }
    )
    if long_only:
        bounds = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        )
    else:
        bounds = None
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    x = optimize.minimize(portfolio_variance(o, cov), w, tol=1e-15, constraints=constraints, bounds=bounds)
    print("Minimum risk portfolio:")
    print(x)
    print()

    r = np.array([
        0.000839294,
        0.000897196,
        0.000402819,
        0.000897710,
        0.000878252,
        0.000746935,
        0.002028144,
        0.000814831,
        0.000625548,
        0.000511312,
    ])
    rf = 0
    x = optimize.minimize(inverse_sharpe_ratio(r, rf, o, cov), w, tol=1e-8, constraints=constraints, bounds=bounds)
    print("Tangent portfolio (best return / risk ratio):")
    print(x)
    mean = r.dot(x.x)
    stdev = math.sqrt(portfolio_variance(o, cov)(x.x))
    sharpe = sharpe_ratio(r, rf, o, cov)(x.x)
    print('Mean:', mean * 100, '%')
    print('Stdev:', stdev * 100, '%')
    print('Sharpe ratio:', sharpe)
    for i in range(len(x.x)):
        print(x.x[i])

if __name__ == '__main__':
    # msft_vs_wfc()
    # all_stocks(True)
    all_stocks()
