import numpy as np
from numpy import convolve
import csv
from scipy.linalg import solve
import math


def multiply_polynomials(polynomials):
    result = [1]

    for polynomial in polynomials:
        result = np.convolve(result, polynomial)

    return np.array(result)


def get_lagrange_polynomial(x, y):
    if len(x) != len(y):
        raise Exception('Wrong data length')

    # Generate all possible monomials
    monomials = [np.array([1, -v]) for v in x]

    # Construct polynomials to the result
    lagrange_polynomials = []
    for i in range(len(x)):
        polynomial = multiply_polynomials(monomials[:i] + monomials[i + 1:])
        lagrange_polynomials.append(polynomial * y[i])

    # Construct divider
    dividers = np.ones(len(y))
    for i in range(len(y)):
        for j in range(len(y)):
            if i != j:
                dividers[i] *= x[i] - x[j]

    dividers = np.reshape(dividers, (-1, 1))

    return np.sum(np.divide(lagrange_polynomials, dividers), axis=0)

def get_polynomial_value(coefficients, x):
    n = len(coefficients)
    y = 0
    for i in range(n):
        y = y + coefficients[n - i - 1] * (x ** i)
    return y


def get_spline_coeffs(x, y):
    functions_count = len(x) - 1
    matrix = [[0 for x in range(functions_count * 4)] for y in range(functions_count * 4)]
    b = []
    empty_row = [0] * 4 * functions_count
    current_row = 0

    # values in given nodes
    for i in range(functions_count):
        matrix[current_row][i * 4] = 1
        b.append(y[i])
        current_row = current_row + 1

        val = x[i + 1] - x[i]

        matrix[current_row][i * 4] = 1
        matrix[current_row][i * 4 + 1] = val
        matrix[current_row][i * 4 + 2] = val ** 2
        matrix[current_row][i * 4 + 3] = val ** 3

        current_row = current_row + 1
        b.append(y[i + 1])

    # intern nodes
    for i in range(1, len(x) - 1):
        # derivatives 1st degree
        val = x[i] - x[i - 1]

        matrix[current_row][(i - 1) * 4 + 1] = 1
        matrix[current_row][(i - 1) * 4 + 2] = 2 * val
        matrix[current_row][(i - 1) * 4 + 3] = 3 * val ** 2

        matrix[current_row][i * 4 + 1] = -1

        current_row = current_row + 1
        b.append(0)

        # derivatives 2nd degree
        matrix[current_row][(i - 1) * 4 + 2] = 2
        matrix[current_row][(i - 1) * 4 + 3] = 6 * val

        matrix[current_row][i * 4 + 2] = -2

        current_row = current_row + 1
        b.append(0)

    # extern nodes
    matrix[current_row][2] = 1  # for min x

    current_row = current_row + 1
    b.append(0)

    value = x[len(x) - 1] - x[len(x) - 2]

    matrix[current_row][(functions_count * 4) - 2] = 2
    matrix[current_row][(functions_count * 4) - 1] = 6 * val
    b.append(0)

    coefficients = solve(matrix, b)

    return coefficients



def get_spline_value(coefficients, x, x0):
    n = len(coefficients)
    y = 0
    for i in range(n):
        y = y + coefficients[n - i - 1] * ((x - x0) ** i)
    return y


def get_interpolated_value(coefficients, x_vals, x):
    coefficients = [coefficients[i:i + 4] for i in range(0, len(coefficients), 4)]
    interval = len(coefficients) - 1

    # Check interval
    for i in range(len(x_vals)):
        if x <= x_vals[i]:
            interval = i - 1
            break

    if interval < 0:
        interval = 0

    return get_spline_value(list(reversed(coefficients[interval])), x, x_vals[interval])


