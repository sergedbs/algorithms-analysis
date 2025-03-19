"""
Fibonacci Number Calculation Module
==================================

This module provides multiple implementations of the Fibonacci sequence calculation
with different time and space complexity characteristics.

The module contains implementations using:
- Recursive approach
- Dynamic programming with memoization
- Bottom-up dynamic programming
- Matrix exponentiation
- Binet's formula (with standard and arbitrary precision)
- Space-optimized iterative approach
- Fast doubling method
"""

from decimal import Decimal, getcontext

import numpy as np


def fib_recursive(n):
    """
    Calculate the nth Fibonacci number using naive recursion.

    :param n: The position in the Fibonacci sequence (0-indexed)
    :type n: int
    :return: The nth Fibonacci number, or -1 if n > 35
    :rtype: int

    :Time Complexity: O(2^n) - Exponential due to redundant calculations
    :Space Complexity: O(n) - Maximum recursion depth

    :Note: Limited to n <= 35 to prevent stack overflow
    """
    if n > 35:
        return -1
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


def fib_memoized(n, memo=None):
    """
    Calculate the nth Fibonacci number using memoization (top-down dynamic programming).

    :param n: The position in the Fibonacci sequence (0-indexed)
    :type n: int
    :param memo: Dictionary to store previously calculated values
    :type memo: dict, optional
    :return: The nth Fibonacci number, or -1 if n > 2000
    :rtype: int

    :Time Complexity: O(n) - Each value calculated only once
    :Space Complexity: O(n) - For storing calculated values

    :Note: Limited to n <= 2000 to prevent excessive memory usage
    """
    if n > 2000:
        return -1
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memoized(n - 1, memo) + fib_memoized(n - 2, memo)
    return memo[n]


def fib_iterative(n):
    """
    Calculate the nth Fibonacci number using bottom-up dynamic programming.

    :param n: The position in the Fibonacci sequence (0-indexed)
    :type n: int
    :return: The nth Fibonacci number
    :rtype: int

    :Time Complexity: O(n) - Linear iteration through values
    :Space Complexity: O(n) - For storing all previous Fibonacci numbers
    """
    if n <= 1:
        return n
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib[n]


def fib_matrix(n):
    """
    Calculate the nth Fibonacci number using matrix exponentiation.

    Uses the property that:
    [1 1]^n   [F(n+1) F(n)]
    [1 0]   = [F(n)   F(n-1)]

    :param n: The position in the Fibonacci sequence (0-indexed)
    :type n: int
    :return: The nth Fibonacci number
    :rtype: int

    :Time Complexity: O(log n) - Due to fast matrix exponentiation
    :Space Complexity: O(1) - Constant space for matrices
    """
    def matrix_mult(a, b):
        """
        Multiply two 2x2 matrices and return the result.

        :param a: First matrix
        :param b: Second matrix
        :return: Matrix product of A and B
        """
        return np.dot(a, b).tolist()

    def matrix_power(matrix, p):
        """
        Compute matrix^p using divide and conquer approach.

        :param matrix: Base matrix
        :param p: Power to raise matrix to
        :return: Matrix raised to power p
        """
        if p == 1:
            return matrix
        if p % 2 == 0:
            half_pow = matrix_power(matrix, p // 2)
            return matrix_mult(half_pow, half_pow)
        else:
            return matrix_mult(matrix, matrix_power(matrix, p - 1))

    base_matrix = [[0, 1], [1, 1]]
    result = matrix_power(base_matrix, n)
    return result[0][1]


def fib_binet(n):
    """
    Calculate the nth Fibonacci number using Binet's formula.

    Binet's formula: F(n) = (φ^n - ψ^n)/√5, where φ = (1+√5)/2 and ψ = (1-√5)/2

    :param n: The position in the Fibonacci sequence (0-indexed)
    :type n: int
    :return: The nth Fibonacci number, or -1 if n > 70
    :rtype: int

    :Time Complexity: O(1) - Constant time operation
    :Space Complexity: O(1) - Constant space

    :Note: Limited to n <= 70 due to floating-point precision issues
    """
    if n > 70:
        return -1
    phi = (1 + np.sqrt(5)) / 2
    psi = (1 - np.sqrt(5)) / 2
    return int((phi ** n - psi ** n) / np.sqrt(5))


def fib_binet_decimal(n):
    """
    Calculate the nth Fibonacci number using Binet's formula with arbitrary precision.

    :param n: The position in the Fibonacci sequence (0-indexed)
    :type n: int
    :return: The nth Fibonacci number, or -1 if n > 1000
    :rtype: int

    :Time Complexity: O(n) - Due to arbitrary precision operations
    :Space Complexity: O(n) - For storing large decimal numbers

    :Note: Limited to n <= 1000 for practical reasons
    """
    if n > 1000:
        return -1
    getcontext().prec = n + 10  # Set precision
    sqrt5 = Decimal(5).sqrt()
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    return int((phi ** n - psi ** n) / sqrt5)


def fib_optimized(n):
    """
    Calculate the nth Fibonacci number using iterative approach with O(1) space.

    :param n: The position in the Fibonacci sequence (0-indexed)
    :type n: int
    :return: The nth Fibonacci number
    :rtype: int

    :Time Complexity: O(n) - Linear iteration
    :Space Complexity: O(1) - Only stores the last two Fibonacci numbers
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fib_fast_doubling(n):
    """
    Calculate the nth Fibonacci number using the fast doubling method.

    Based on the recurrence relations:
    F(2k) = F(k) * [2*F(k+1) - F(k)]
    F(2k+1) = F(k+1)^2 + F(k)^2

    :param n: The position in the Fibonacci sequence (0-indexed)
    :type n: int
    :return: The nth Fibonacci number
    :rtype: int

    :Time Complexity: O(log n) - Logarithmic time due to recursive doubling
    :Space Complexity: O(log n) - Recursion depth
    """
    def fib_doubling(k):
        """
        Helper recursive function that returns F(k) and F(k+1).

        :param k: Position in Fibonacci sequence
        :return: Tuple of (F(k), F(k+1))
        """
        if k == 0:
            return 0, 1
        else:
            a, b = fib_doubling(k // 2)
            c = a * ((b << 1) - a)  # F(2k) = F(k) * [2*F(k+1) - F(k)]
            d = a * a + b * b       # F(2k+1) = F(k+1)^2 + F(k)^2
            if k & 1:
                return d, c + d
            else:
                return c, d

    return fib_doubling(n)[0]