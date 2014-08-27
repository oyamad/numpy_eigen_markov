"""
Filename: unittest_numpy_eigen_markov.py
Author: Daisuke Oyama

Unittest for stoch_eig

Input Markov matrices defined by the Kandori-Mailath-Rob model with
- two actions (0 and 1),
- payoffs being characterized by the level of p-dominance of action 1,
- N players, and
- mutation probability epsilon.

References
----------

    https://github.com/oyamad/test_mc_compute_stationary

"""

from __future__ import division

import sys
import random
import argparse
import numpy as np
from scipy.stats import binom
import unittest
from eigen_markov import stoch_eig


# Default parameter values
default_N, default_epsilon = 27, 1e-2
default_p = 1/3  # action 1 is risk-dominant

# Default move: 'sequential' or 'simultaneous'
default_move = 'sequential'

# Tolerance level
default_TOL = 0


def KMR_Markov_matrix_simultaneous(N, p, epsilon):
    """
    Generate the Markov matrix for the KMR model with *simultaneous* move

    Parameters
    ----------
    N : int
        Number of players

    p : float
        Level of p-dominance of action 1, i.e.,
        the value of p such that action 1 is the BR for (1-q, q) for any q > p,
        where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)

    epsilon : float
        Probability of mutation

    Returns
    -------
    P : numpy.ndarray
        Markov matrix for the KMR model with simultaneous move

    Notes
    -----
    For simplicity, the transition probabilities are computed under the assumption
    that a player is allowed to be matched to play with himself.

    """
    P = np.empty((N+1, N+1), dtype=float)
    for n in range(N+1):
        P[n, :] = \
            (n/N < p) * binom.pmf(range(N+1), N, epsilon/2) + \
            (n/N == p) * binom.pmf(range(N+1), N, 1/2) + \
            (n/N > p) * binom.pmf(range(N+1), N, 1-epsilon/2)
    return P


def KMR_Markov_matrix_sequential(N, p, epsilon):
    """
    Generate the Markov matrix for the KMR model with *sequential* move

    Parameters
    ----------
    N : int
        Number of players

    p : float
        Level of p-dominance of action 1, i.e.,
        the value of p such that action 1 is the BR for (1-q, q) for any q > p,
        where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)

    epsilon : float
        Probability of mutation

    Returns
    -------
    P : numpy.ndarray
        Markov matrix for the KMR model with simultaneous move

    """
    P = np.zeros((N+1, N+1), dtype=float)
    P[0, 0], P[0, 1] = 1 - epsilon * (1/2), epsilon * (1/2)
    for n in range(1, N):
        P[n, n-1] = \
            (n/N) * (epsilon * (1/2) +
                     (1 - epsilon) * (((n-1)/(N-1) < p) + ((n-1)/(N-1) == p) * (1/2))
                     )
        P[n, n+1] = \
            ((N-n)/N) * (epsilon * (1/2) +
                         (1 - epsilon) * ((n/(N-1) > p) + (n/(N-1) == p) * (1/2))
                         )
        P[n, n] = 1 - P[n, n-1] - P[n, n+1]
    P[N, N-1], P[N, N] = epsilon * (1/2), 1 - epsilon * (1/2)
    return P


class TestComputeStationary(unittest.TestCase):
    def setUp(self):
        self.P, self.v = P, v

    def test_markov_matrix(self):
        for i in range(len(self.P)):
            self.assertEqual(sum(self.P[i, :]), 1)

    def test_sum_one(self):
        self.assertTrue(np.allclose(sum(self.v), 1, atol=TOL))

    def test_nonnegative(self):
        self.assertEqual(np.prod(self.v >= 0-TOL), 1)

    def test_left_eigen_vec(self):
        self.assertTrue(np.allclose(np.dot(self.v, self.P), self.v, atol=TOL))

    def tearDown(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unittest for mc_compute_stationary.')
    parser.add_argument(
        "--N", dest='N', type=int, action='store', default=default_N,
        metavar='N', help=u'N = number of players'
        )
    parser.add_argument(
        "--epsilon", dest='epsilon', type=float, action='store', default=default_epsilon,
        metavar='epsilon', help=u'epsilon = mutation probability'
        )
    parser.add_argument(
        "--p", dest='p', action='store', default=default_p,
        metavar='p', help=u'p = level of p-dominance of action 1'
        )
    parser.add_argument(
        "--tolerance", type=float, dest='tol', action='store', default=default_TOL,
        metavar='tol', help=u'tol = tolerance'
        )
    parser.add_argument(
        "--move", dest='move', action='store', default=default_move,
        help=u'\'sequential\' (default) or \'simulataneous\''
        )
    args = parser.parse_args()

    if args.move == 'simultaneous':
        P = KMR_Markov_matrix_simultaneous(N=args.N, p=args.p, epsilon=args.epsilon)
    else:
        P = KMR_Markov_matrix_sequential(N=args.N, p=args.p, epsilon=args.epsilon)

    v = stoch_eig(P)  # Stochastic eigenvector of P

    TOL = args.tol

    print 'N =', args.N, ', epsilon =', args.epsilon, '\n'

    if args.N <= 5:
        print 'P =\n', P, '\n'
    print 'v =\n', v, '\n'
    print 'TOL =', TOL, '\n'

    suite = unittest.TestLoader().loadTestsFromTestCase(TestComputeStationary)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
