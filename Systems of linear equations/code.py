import math
import time
import matplotlib.pyplot as plt

def display_matrix(matrix, size):
    for i in range(size):
        for j in range(size):
            print(matrix[i][j], end=' ')
        print('\n')


def create_matrix(N, a1, a2, a3):
    rows, cols = (N, N)
    A = [[0.0 for i in range(cols)] for j in range(rows)]

    for i in range(N):
        A[i][i] = a1
        if i + 1 < N:
            A[i][i+1] = a2
        if i + 2 < N:
            A[i][i+2] = a3

        if i - 1 >= 0:
            A[i][i-1] = a2
        if i - 2 >= 0:
            A[i][i-2] = a3
    return A


def get_vector_norm_euklides(v):
    N = len(v)
    ret = 0
    for i in range(N):
        ret += v[i]*v[i]
    return math.sqrt(ret)


def multiply_matrix(A, x):
    N = len(x)
    ret = [0.0 for i in range(N)]

    for i in range(N):
        sum = 0.0
        for j in range(N):
            sum += A[i][j] * x[j]
        ret[i] = sum
    return ret


def substract_vectors(a, b):
    N = len(a)
    for i in range(N):
        a[i] -= b[i]
    return a


# Ax = b
def get_vector_residuum(A, x, b):
    N = len(x)
    Ax = multiply_matrix(A, x) #wektor o rozmiarze N
    residuum = substract_vectors(Ax, b) # wektor o rozmiarze N
    return residuum


def get_vector_norm(A, x, b):
    N = len(x)
    residuum = get_vector_residuum(A, x, b)
    norm = get_vector_norm_euklides(residuum)
    #print("norma = " + str(norm))
    return norm


def solve_jacobi(A, b, epsilon):
    N = len(b)
    x = [0.0 for i in range(N)]  # macierz X, na wynik
    x_prev = [1.0 for i in range(N)]  # początkowe "wyniki"

    k = 0
    # print("epsilon: " + str(epsilon))
    start = time.time()

    while get_vector_norm(A, x, b) > epsilon:
        for i in range(N):
            x[i] = b[i]

            for j in range(N):
                if i != j:
                    x[i] -= A[i][j] * x_prev[j]
            x[i] /= A[i][i]

        x_prev = x.copy()
        k += 1
    # end = time.time()
    # print("czas wykonania: " + str(end - start) + "s")
    # print("liczba iteracji: " + str(k))
    return x


def solve_gauss_seidel(A, b, epsilon):
    N = len(b)
    x = [1.0 for i in range(N)]
    x_prev = [1.0 for i in range(N)]

    k = 0
    # print("epsilon: " + str(epsilon))
    start = time.time()

    while get_vector_norm(A, x, b) > epsilon:
        for i in range(N):
            triangle_sum = 0

            # macierz trójkątna górna
            for j in range(i):
                triangle_sum += A[i][j] * x[j]

            # macierz trójkątna dolna
            for j in range(i + 1, N):
                triangle_sum += A[i][j] * x_prev[j]

            x[i] = (b[i] - triangle_sum) / A[i][i]
            x_prev = x.copy()
        k += 1

    end = time.time()
    # print("czas wykonania: " + str(end - start) + "s")
    # print("liczba iteracji: " + str(k))
    return x


def solve_LU_factor(A, b):
    N = len(b)
    x = [1.0 for i in range(N)]

    L = A.copy()
    U = create_matrix(N, 1, 0, 0)

    start = time.time()

    # L*U*x = b
    # print("liczenie L i U")
    for i in range(N - 1):
        for j in range(i + 1, N):
            L[j][i] = U[j][i] / U[i][i]

            for k in range(i, N):
                U[j][k] = U[j][k] - L[j][i] * U[i][k]

    # Ly = b, macierz trójkątna górna, podstawianie wprzód
    # print("trójkąt góra")
    y = [0.0 for i in range(N)]

    for i in range(N):
        Ly = 0
        for j in range(i):
            Ly += L[i][j] * y[j]
        y[i] = (b[i] - Ly) / L[i][i]

    # Ux = y, macierz trójkątna dolna, podstawianie wstecz
    # print("trójkąt dół")
    for i in reversed(range(0, N - 1)):
        Ux = 0
        for j in range(i + 1, N):
            Ux += U[i][j] * x[j]
        x[i] = (y[i] - Ux) / U[i][i]

    end = time.time()
    # print("czas wykonania: " + str(end - start) + "s")
    get_vector_norm(A, x, b)
    return x

# zad A
e = 9.0
N = 915
a1 = 5.0 + e
a2 = -1.0
a3 = -1.0

A = create_matrix(N, a1, a2, a3)
b = [0 for i in range(N)]
f = 4

for i in range(N):
    b[i] = math.sin(i*(f + 1))

# zad B
jac = solve_jacobi(A, b, pow(10, -9))
gau = solve_gauss_seidel(A, b, pow(10, -9))

# zad C
A = create_matrix(N, 3, -1, -1)
jac = solve_jacobi(A, b, pow(10, -9))
gau = solve_gauss_seidel(A, b, pow(10, -9))

# zad D
solve_LU_factor(A, b)

# zad E
it = [100, 500, 1000, 2000, 3000]
jacobi_time =  [0.0 for i in range(5)]
gauss_time =  [0.0 for i in range(5)]
LU_time =  [0.0 for i in range(5)]

epsilon = pow(10, -9)

# jacobi
i = 0
for n in it:
    A = create_matrix(n, a1, a2, a3)
    b = [1.0 for i in range(n)]

    start = time.time()
    solve_jacobi(A, b, epsilon)
    end = time.time()
    jacobi_time[i] = end - start

    print(jacobi_time[i])

    i += 1

plt.plot(it, jacobi_time, 'ro')
plt.title('Czas trwa algorytmu Jacobiego w zależności od liczby niewiadomych')
plt.ylabel('czas obliczeń [s]')
plt.xlabel('liczba niewiadomych')
plt.show()

# gauss-seidel
i = 0
for n in it:
    A = create_matrix(n, a1, a2, a3)
    b = [1.0 for i in range(n)]

    start = time.time()
    solve_gauss_seidel(A, b, epsilon)
    end = time.time()
    gauss_time[i] = end - start

    print(gauss_time[i])

    i += 1

plt.plot(it, gauss_time, 'ro')
plt.title('Czas trwa algorytmu Gaussa-Seidla w zależności od liczby niewiadomych')
plt.ylabel('czas obliczeń [s]')
plt.xlabel('liczba niewiadomych')
plt.show()

# LU
i = 0
for n in it:
    A = create_matrix(n, a1, a2, a3)
    b = [1.0 for i in range(n)]

    start = time.time()
    solve_LU_factor(A, b)
    end = time.time()
    LU_time[i] = end - start

    print(LU_time[i])

    i += 1

plt.plot(it, LU_time, 'ro')
plt.title('Czas trwa algorytmu faktoryzacji LU w zależności od liczby niewiadomych')
plt.ylabel('czas obliczeń [s]')
plt.xlabel('liczba niewiadomych')
plt.show()