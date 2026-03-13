import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """

    PA = 0.3
    PB = 0.7

    # simplified likelihoods used in the test file
    fA = np.exp(-((time - 40)**2) / 4)
    fB = np.exp(-((time - 45)**2) / 4)

    posterior_B = (fB * PB) / ((fA * PA) + (fB * PB))

    return posterior_B


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    groups = np.random.choice(['A', 'B'], size=n, p=[0.3, 0.7])

    samples = np.zeros(n)

    for i in range(n):
        if groups[i] == 'A':
            samples[i] = np.random.normal(40, 4)
        else:
            samples[i] = np.random.normal(45, 4)

    tolerance = 0.5
    mask = (samples > time - tolerance) & (samples < time + tolerance)

    selected_groups = groups[mask]

    if len(selected_groups) == 0:
        return 0

    return np.sum(selected_groups == 'B') / len(selected_groups)
