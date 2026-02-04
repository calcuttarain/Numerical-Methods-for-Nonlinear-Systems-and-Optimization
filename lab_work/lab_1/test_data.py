import numpy as np

# A nu e patratica
A1 = np.array([
    [3, 5, 3],
    [2, 2, 3]
])
b1 = np.array([1, 3])

# A si b sunt incompatibili
A2 = np.array([
    [3, 3, -6],
    [-4, 7, -8],
    [5, 7, -9]
])
b2 = np.array([0, -5, 3])

# A singulara
A3 = np.array([
    [4, 1, 1],
    [2, -9, 0],
    [0, -6, -8]
])
b3 = np.array([6, -7, -14])

# solutie unica, metode iterative divergente
A4 = np.array([
    [3, 0, 4],
    [7, 4, 2],
    [-1, 1, 2]
])
b4 = np.array([7, 13, 2])

# solutie unica, metode iterative convergente
A5 = np.array([
    [3, -1, 0],
    [-1, 3, -1],
    [0, -1, 3]
])
b5 = np.array([2, 1, 2])

tests = [
    (A1, b1),
    (A2, b2),
    (A3, b3),
    (A4, b4),
    (A5, b5)
]
