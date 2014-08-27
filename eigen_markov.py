"""
Filename: eigen_markov.py

Author: Daisuke Oyama

This file contains some numpy-based routines specialized for stochastic
(Markov) matrices.

stoch_eig : returns the stochastic eigenvector (stationary distribution)
    of an irreducible stochastic matrix P, i.e., the stochastic vector x
    with x P = x
gth_solve : returns the (normalized) nontrivial solution to x A = 0 for
    an irreducible transition rate matrix A


Stochastic matrices
...................

The routine ``stoch_eig`` returns the stochastic eigenvector of an
irreducible stochastic matrix *P*, i.e., the stochastic vector `x` with
`x P = x` (a stochastic matrix, or Markov transition matrix, is a real
square matrix whose entries are nonnegative and rows sum to one).
Internally, the routine passes the input to the ``gth_solve`` routine.

The routine ``gth_solve`` solves for a (normalized) nontrivial solution
to `x A = 0` for an irreducible transition rate matrix *A* (a transition
rate matrix is a real square matrix whose off-diagonal entries are
nonnegative and rows sum to zero), by using the Grassmann-Taksar-Heyman
(GTH) algorithm [1]_, a numerically stable variant of Gaussian
elimination.


Notes
-----
For stochastic matrices, see en.wikipedia.org/wiki/Stochastic_matrix
For transition rate matrices, see en.wikipedia.org/wiki/Transition_rate_matrix
For irreducible matrices, see
en.wikipedia.org/wiki/Perron-Frobenius_theorem#Classification_of_matrices

If the input matrix is not irreducible, ``stoch_eig`` (``gth_solve``)
returns the solution corresponding to the irreducible class of indices
that contains the first recurrent index.

For the GTH algorithm, see the excellent lecture notes [2]_.

References
----------
.. [1] W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative
   Analysis and Steady State Distributions for Markov Chains,"
   Operations Research (1985), 1107-1116.
.. [2] W. J. Stewart, "Performance Modelling and Markov Chains,"
   www.sti.uniurb.it/events/sfm07pe/slides/Stewart_1.pdf

"""
import numpy as np

try:
    xrange
except:  # python3
    xrange = range


def stoch_eig(P, overwrite=False):
    r"""
    This routine returns the stochastic eigenvector (stationary
    probability distribution vector) of an irreducible stochastic matrix
    *P*, i.e., the solution to `x P = x`, normalized so that its 1-norm
    equals one. Internally, the routine passes the input to the
    ``gth_solve`` routine.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        Stochastic matrix. Must be of shape n x n.
    overwrite : bool, optional(default=False)
        Whether to overwrite P; may improve performance.

    Returns
    -------
    x : numpy.ndarray(float, ndim=1)
        Stochastic eigenvalue (stationary distribution) of P, i.e., the
        solution to x P = x, normalized so that its 1-norm equals one.

    Examples
    --------
    >>> import numpy as np
    >>> from eigen_markov import stoch_eig
    >>> P = np.array([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]])
    >>> x = stoch_eig(P)
    >>> print x
    [ 0.625   0.3125  0.0625]
    >>> print np.dot(x, P)
    [ 0.625   0.3125  0.0625]

    """
    # In fact, stoch_eig, which for the user is a routine to solve
    # x P = x, or x (P - I) = 0, is just another name of the function
    # gth_solve, which solves x A = 0, where the GTH algorithm,
    # the algorithm used there, does not use the actual values of
    # the diagonals of A, under the assumption that
    # A_{ii} = \sum_{j \neq i} A_{ij}, and therefore,
    # gth_solve(P-I) = gth_solve(P), so that it is irrelevant whether to
    # pass P or P - I to gth_solve.
    return gth_solve(P, overwrite=overwrite)


def gth_solve(A, overwrite=False):
    r"""
    This routine computes a nontrivial solution of a linear equation
    system of the form `x A = 0`, where *A* is an irreducible transition
    rate matrix, by using the Grassmann-Taksar-Heyman (GTH) algorithm, a
    variant of Gaussian elimination. The solution is normalized so that
    its 1-norm equals one.

    Parameters
    ----------
    A : array_like(float, ndim=2)
        Transition rate matrix. Must be of shape n x n.
    overwrite : bool, optional(default=False)
        Whether to overwrite A; may improve performance.

    Returns
    -------
    x : numpy.ndarray(float, ndim=1)
        Non-zero solution to x A = 0, normalized so that its 1-norm
        equals one.

    Examples
    --------
    >>> import numpy as np
    >>> from eigen_markov import gth_solve
    >>> A = np.array([[-0.1, 0.075, 0.025], [0.15, -0.2, 0.05], [0.25, 0.25, -0.5]])
    >>> x = gth_solve(A)
    >>> print x
    [ 0.625   0.3125  0.0625]
    >>> print np.dot(x, A)
    [ 0.  0.  0.]

    """
    A1 = np.array(A, copy=not overwrite)

    n, m = A1.shape

    if n != m:
        raise ValueError('matrix must be square')

    x = np.zeros(n)

    # === Reduction === #
    for i in xrange(n-1):
        scale = np.sum(A1[i, i+1:n])
        if scale <= 0:
            # Only consider the leading principal minor of size i+1,
            # which is irreducible
            n = i+1
            break
        A1[i+1:n, i] /= scale

        for j in xrange(i+1, n):
            A1[i+1:n, j] += A1[i, j] * A1[i+1:n, i]

    # === Backward substitution === #
    x[n-1] = 1
    for i in xrange(n-2, -1, -1):
        x[i] = np.sum((x[j] * A1[j, i] for j in xrange(i+1, n)))

    # === Normalization === #
    x /= np.sum(x)

    return x
