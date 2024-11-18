# 2024 (c) Hanc, Hancova (jozef.hanc@upjs.sk)
# Ver.: 11-November-2024 
# 
"""
Original code for barycentric PDF interpolation based on Chebyshev polynomials: Witkovský, V. 2023. CharFunTool. https://github.com/witkovsky/CharFunTool.
Python conversion and modification: Jozef Hanc, Martina Hancova, https://github.com/JupyterPER
"""


import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from numba import njit, jit, vectorize, float64, int64, prange

def chebyshev_interpolant(x_cheb, f_cheb):
    """
    Create a Chebyshev interpolant for a function based on Chebyshev nodes and values.

    Parameters:
    - x_cheb: numpy array of Chebyshev nodes (assumed to be in the interval [-1, 1])
    - f_cheb: numpy array of function values at the Chebyshev nodes
    - deg: Degree of the Chebyshev polynomial to be used for interpolation
    
    Returns:
    - interpolant: A function that takes an array of points and returns interpolated values.
    """
    # Compute Chebyshev coefficients using NumPy's chebfit function
    coeffs = chebfit(x_cheb, f_cheb, deg)

    # Define the interpolant function using the Chebyshev coefficients
    def interpolant(x):
        return chebval(x, coeffs)

    return interpolant

def chebyshev_points(a, b, n_points):
    """
    Generate n_points Chebyshev nodes in the interval [a, b].

    Parameters:
    - a: Start of the interval
    - b: End of the interval
    - n_points: Number of Chebyshev nodes to generate

    Returns:
    - numpy array of Chebyshev nodes in the interval [a, b]
    """
    return (b - a) * (-np.cos(np.pi * np.arange(n_points) / (n_points - 1)) + 1) / 2 + a

@njit(parallel=True, fastmath=True)
def chebyshev_points_n(a, b, n_points):
    """
    Generate n_points Chebyshev nodes in the interval [a, b] using Numba parallelization.

    Parameters:
    - a: float
        Start of the interval
    - b: float
        End of the interval
    - n_points: int
        Number of Chebyshev nodes to generate

    Returns:
    - points: numpy array
        Chebyshev nodes in the interval [a, b]
    """
    points = np.empty(n_points, dtype=np.float64)
    n_minus_one = n_points - 1
    for k in prange(n_points):
        theta = np.pi * k / n_minus_one
        x = np.cos(theta)
        points[k] = (b - a) * (-x + 1) / 2 + a
    return points

import numpy as np
from scipy.interpolate import BarycentricInterpolator

def InterpPDF(xNew, xGiven, pdfGiven=None):
    '''
    Evaluates the interpolant for the PDF at specified values xNew,
    calculated from the pre-calculated (known) values xGiven and pdfGiven.
    The evaluation is based on barycentric interpolation.

    Parameters:
        xNew: array_like
            The x-values where the PDF is evaluated.
        xGiven: array_like or dict
            If array_like, these are the x-values where the PDF is known.
            If dict (e.g., result from a computation), it should contain 'x' and 'pdf'.
        pdfGiven: array_like, optional
            The known PDF values at xGiven.

    Returns:
        pdf: array_like
            The interpolated PDF values at xNew.
    '''

    # If pdfGiven is None, extract xGiven and pdfGiven from xGiven (assumed to be a dict)
    if pdfGiven is None:
        if isinstance(xGiven, dict) and 'x' in xGiven and 'pdf' in xGiven:
            result = xGiven
            xGiven = result['x']
            pdfGiven = result['pdf']
        else:
            raise ValueError('Missing inputs: pdfGiven is None and xGiven is not a dict with "x" and "pdf" keys.')

    # Ensure inputs are numpy arrays
    xNew = np.asarray(xNew)
    szx = xNew.shape
    xNew_flat = xNew.flatten()
    xGiven = np.asarray(xGiven)
    pdfGiven = np.asarray(pdfGiven)

    # Identify xNew within the range of xGiven
    xGiven_min = np.min(xGiven)
    xGiven_max = np.max(xGiven)
    idx = (xNew_flat >= xGiven_min) & (xNew_flat <= xGiven_max)

    # Initialize pdf to zeros
    pdf = np.zeros_like(xNew_flat)

    if np.any(idx):
        # Create barycentric interpolator
        interpolator = BarycentricInterpolator(xGiven, pdfGiven)
        # Compute pdf at xNew where idx is True
        pdf[idx] = interpolator(xNew_flat[idx])

    # Ensure pdf is non-negative
    pdf = np.maximum(0, pdf)

    # Reshape pdf to the original shape of xNew
    pdf = pdf.reshape(szx)

    return pdf

def DE_CC_pdf(f, x, a, b, chebyPts, rtol):
    """
    Computes the interpolated PDF using Chebyshev points.

    Parameters:
        f: function
            The function to evaluate at Chebyshev points.
        x: array_like
            The x-values where the PDF is evaluated.
        a: float
            The lower bound of the interval.
        b: float
            The upper bound of the interval.
        chebyPts: int
            The number of Chebyshev points.
        rtol: float
            The relative tolerance for function evaluation.

    Returns:
        f_eval: array_like
            The interpolated PDF values at x.
    """
    xN = chebyPts
    x1 = chebyshev_points(a, b, xN)  # Get Chebyshev points in the interval [a, b]
    f1 = f(x1, rtol)  # Evaluate function f at Chebyshev points
    f_eval = InterpPDF(x, x1, f1)  # Interpolate the PDF at given x with barycentric interpolation.
    return f_eval
