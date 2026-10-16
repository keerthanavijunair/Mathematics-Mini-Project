#the code implements different methods for solving systems of linear equations using NumPy.
import numpy as np


def gaussian_elimination(A, b):
    #converts A & b into floating point copies, later does back substitution to find x
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    n = len(b) #stores number of equations in the system.

    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        #finds the row with the largest absolute value in column i starting from current row (partial pivoting).

        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
            #performs swapping if the best pivot is in another row.

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            #visits each row after pivot and calculate the multiplier used to eliminate the current column entry.

            A[j, i:] = A[j, i:] - factor * A[i, i:]
            b[j] = b[j] - factor * b[i]
            #make elements below pivot as zero and apply the same for b.

    x = np.zeros(n)

    for i in range(n - 1, -1, -1): #back substitution
        x[i] = (
            b[i] - np.dot(A[i, i + 1:], x[i + 1:]) #np.dot computes the dot product
        ) / A[i, i]

    return x


def rref(A):
    #converts the matrix into floating-point form, identifies its number of rows and columns
    A = A.astype(float).copy()

    rows, cols = A.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        max_row = pivot_row + np.argmax(
            np.abs(A[pivot_row:, col])
        )
        #selects row with largest absolute pivot
        if abs(A[max_row, col]) < 1e-12: #skips columns whose pivot smaller than 10^-12, swap if required
            continue

        A[[pivot_row, max_row]] = A[[max_row, pivot_row]]

        A[pivot_row] = (
            A[pivot_row] / A[pivot_row, col]
        )
        #making pivots equal to 1.

        for row in range(rows):
            if row != pivot_row:
                factor = A[row, col]
                A[row] = A[row] - factor * A[pivot_row]
                #make all other entries in the pivot column zero.

        pivot_row += 1

    return A


def rref_solve(A, b):
    augmented = np.column_stack((A, b))
    reduced = rref(augmented)

    return reduced[:, -1]


def lu_decomposition(A):
    #factorizes A into L and U
    A = A.astype(float).copy()

    n = A.shape[0]

    L = np.eye(n)
    U = A.copy()

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]

            L[j, i] = factor

            U[j, i:] = (
                U[j, i:] - factor * U[i, i:]
            )

    return L, U


def forward_substitution(L, b): #solves lower trinagular system Ly=b.
    n = len(b)

    y = np.zeros(n)

    for i in range(n):
        y[i] = (
            b[i] - np.dot(L[i, :i], y[:i])
        ) / L[i, i]

    return y


def back_substitution(U, y): #solves upper triangular system Ux=y.
    n = len(y)

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            y[i] - np.dot(U[i, i + 1:], x[i + 1:])
        ) / U[i, i]

    return x


def lu_solve(A, b): #solve Ax=b using LU decomposition.
    L, U = lu_decomposition(A)

    y = forward_substitution(L, b)
    x = back_substitution(U, y)

    return x, L, U


