import numpy as np
import matplotlib.pyplot as plt
import os

from test_data import tests

np.random.seed(10)

ITMAX = 1000
TOL = 1e-14


def jor_method(A: np.ndarray, b: np.ndarray, x_0: np.ndarray, omega: float, itmax: int, tol: float, opt: int = 0) -> dict:
    n_row, n_col = A.shape

    k_f = 0
    converged = False
    errors = []

    x_k = x_0.copy()
    x_ant = x_0.copy()

    while True:
        # compute x_k
        x_new = np.zeros_like(x_k)
        for i in range (n_row):
            l_u = np.sum([A[i][j] * x_ant[j] for j in range(n_col) if i != j])
            x_new[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]
        x_k = x_new.copy()

        # compute correction/residual
        r_k = np.linalg.norm(b - A @ x_k)
        d_k = np.linalg.norm(x_k - x_ant)

        if opt == 0:
            chosen_err_crit = d_k 
        else:
            chosen_err_crit = r_k 

        errors.append((np.round(d_k, 4), np.round(r_k, 4)))

        if chosen_err_crit < tol or k_f >= itmax:
            converged = True
            break

        k_f += 1
        x_ant = x_k.copy()

    result = {'x': x_k, 'k_f': k_f, 'converged': converged, 'errors': errors}

    return result



def sor_method(A: np.ndarray, b: np.ndarray, x_0: np.ndarray, omega: float, itmax: int, tol: float, opt: int = 0, order = 'backward') -> dict:
    n_row, n_col = A.shape

    k_f = 0
    converged = False
    errors = []

    x_k = x_0.copy()
    x_ant = x_0.copy()

    while True:
        # compute x_k
        x_new = np.zeros_like(x_k)
        if order == 'backward':
            for i in range (n_row - 1, -1, -1):
                # singura schimbare fata de Jacobi e ca inlocuim x_ant cu x_k in suma
                l_u = np.sum([A[i][j] * x_k[j] for j in range(n_col) if i != j])
                x_new[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]

        else:
            for i in range (n_row):
                # singura schimbare fata de Jacobi e ca inlocuim x_ant cu x_k in suma
                l_u = np.sum([A[i][j] * x_k[j] for j in range(n_col) if i != j])
                x_new[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]

        x_k = x_new.copy()

        # compute correction/residual
        r_k = np.linalg.norm(b - A @ x_k)
        d_k = np.linalg.norm(x_k - x_ant)

        if opt == 0:
            chosen_err_crit = d_k 
        else:
            chosen_err_crit = r_k 

        errors.append((np.round(d_k, 4), np.round(r_k, 4)))

        if chosen_err_crit < tol or k_f >= itmax:
            converged = True
            break

        k_f += 1
        x_ant = x_k.copy()

    result = {'x': x_k, 'k_f': k_f, 'converged': converged, 'errors': errors}

    return result



def ssor_method(A: np.ndarray, b: np.ndarray, x_0: np.ndarray, omega: float, itmax: int, tol: float, opt: int = 0, order = 'backward') -> dict:
    n_row, n_col = A.shape

    k_f = 0
    converged = False
    errors = []

    x_k = x_0.copy()
    x_ant = x_0.copy()

    while True:
        # compute x_k
        x_new = np.zeros_like(x_k)
        # ascending
        for i in range (n_row):
            # singura schimbare fata de Jacobi e ca inlocuim x_ant cu x_k in suma
            l_u = np.sum([A[i][j] * x_k[j] for j in range(n_col) if i != j])
            x_new[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]
        # backward
        for i in range (n_row - 1, -1, -1):
            # singura schimbare fata de Jacobi e ca inlocuim x_ant cu x_k in suma
            l_u = np.sum([A[i][j] * x_k[j] for j in range(n_col) if i != j])
            x_new[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]
        x_k = x_new.copy()

        # compute correction/residual
        r_k = np.linalg.norm(b - A @ x_k)
        d_k = np.linalg.norm(x_k - x_ant)

        if opt == 0:
            chosen_err_crit = d_k 
        else:
            chosen_err_crit = r_k 

        errors.append((np.round(d_k, 4), np.round(r_k, 4)))

        if chosen_err_crit < tol or k_f >= itmax:
            converged = True
            break

        k_f += 1
        x_ant = x_k.copy()

    result = {'x': x_k, 'k_f': k_f, 'converged': converged, 'errors': errors}

    return result



def plot_errors(errors, method: str, num_test: int, omega: float, test: bool = True, save: bool = False):
    d_arr = [error[0] for error in errors]
    r_arr = [error[1] for error in errors]
    iterations = range(len(errors))

    fig, axs = plt.subplots(1, 2, figsize = (16, 9))

    axs[0].plot(iterations, d_arr, 'r-')
    axs[0].set_title("Correction norm: " + r"$\|x^{(k)} - x^{(k-1)}\|\,$")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel(r"$\|\delta x\|$")
    axs[0].set_yscale('log')
    axs[0].grid(True)

    axs[1].plot(iterations, r_arr, 'r-')
    axs[1].set_title("Residual norm: " + r"$\|b - A x^{(k)}\|\,$")
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel(r"$\|r\|$")
    axs[1].set_yscale('log')
    axs[1].grid(True)

    fig.suptitle(f"Test {num_test} for the {method} method, " + r"$\omega$" + f" = {omega}")
    
    if test:
        plt.show()

    if save:
        folder_path = f'./plots/{method}/'
        os.makedirs(folder_path, exist_ok=True)

        filename = f'{num_test}_{omega}'
        plt.savefig(folder_path + filename + '.pdf')




def run_save_all_tests():
    for i, test in enumerate(tests, start = 1):
        A, b = test
        _, n_col = A.shape

        x_0 = np.random.randn(n_col)


        # jacobi method
        omega = 1
        result = jor_method(A, b, x_0, omega, ITMAX, TOL, 0)
        plot_errors(result['errors'], 'Jacobi', i, omega, test = False, save = True)


        # jor method
        omega = 0.9
        result = jor_method(A, b, x_0, omega, ITMAX, TOL, 0)
        plot_errors(result['errors'], 'Jacobi Over-Relaxed', i, omega, test = False, save = True)


        # gauss-seidel method
        omega = 1
        result = sor_method(A, b, x_0, omega, ITMAX, TOL, 0)
        plot_errors(result['errors'], 'Gauss-Seidel', i, omega, test = False, save = True)


        # sor method
        omega = 0.5 
        result = sor_method(A, b, x_0, omega, ITMAX, TOL, 0)
        plot_errors(result['errors'], 'Successive Over-Relaxation', i, omega, test = False, save = True)


        # symmetric gauss-seidel method
        omega = 1
        result = ssor_method(A, b, x_0, omega, ITMAX, TOL, 0)
        plot_errors(result['errors'], 'Symmetric Gauss-Seidel', i, omega, test = False, save = True)


        # ssor method
        omega = 1.5 
        result = ssor_method(A, b, x_0, omega, ITMAX, TOL, 0)
        plot_errors(result['errors'], 'Symmetric Successive Over-Relaxation', i, omega, test = False, save = True)

    plt.close('all')

   

def main():
    run_save_all_tests()

    # demo
    test_num = 3 
    method = 'SSOR'

    omega = 0.8
    opt = 1

    A, b = tests[test_num - 1]

    _, n_col = A.shape
    x_0 = np.random.randn(n_col)

    result = ssor_method(A, b, x_0, omega, ITMAX, TOL, opt)

    print(('-' * 10) + f' Test {test_num}, {method} method, omega = {omega} ' + ('-' * 10))
    if result['converged']:
        print(f'Converged in {result['k_f']} iterations.')
    else:
        print('Not converged.')

    plot_errors(result['errors'], method, test_num, omega, test = True, save = False)



if __name__ == "__main__":
    main()
