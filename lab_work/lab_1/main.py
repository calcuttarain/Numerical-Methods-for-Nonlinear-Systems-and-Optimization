import numpy as np
import matplotlib.pyplot as plt
import os

from test_data import tests

np.random.seed(10)

ITMAX = 1000
TOL = 1e-14


# jacobi, jacobi over-relaxed solution
def jor_solution(A:np.ndarray, b: np.ndarray, x_k: np.ndarray, x_ant: np.ndarray, omega: float) -> np.ndarray:
    n_row, n_col = A.shape

    x_new = np.zeros_like(x_k)

    for i in range (n_row):
        l_u = np.sum([A[i][j] * x_ant[j] for j in range(n_col) if i != j])
        x_new[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]

    return x_new


# gauss-seidel ascending, successive over-relaxation ascending solutions
def sora_solution(A:np.ndarray, b: np.ndarray, x_k: np.ndarray, x_ant: np.ndarray, omega: float) -> np.ndarray:
    n_row, n_col = A.shape

    for i in range (n_row):
    # singura schimbare fata de Jacobi e ca inlocuim x_ant cu x_k in suma
        l_u = np.sum([A[i][j] * x_k[j] for j in range(n_col) if i != j])
        x_k[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]

    return x_k


# gauss-seidel backwards, successive over-relaxation backwards solutions
def sorb_solution(A:np.ndarray, b: np.ndarray, x_k: np.ndarray, x_ant: np.ndarray, omega: float) -> np.ndarray:
    n_row, n_col = A.shape

    for i in range (n_row - 1, -1, -1):
        l_u = np.sum([A[i][j] * x_k[j] for j in range(n_col) if i != j])
        x_k[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]

    return x_k


# symmetric successive over-relaxation solution
def ssor_solution(A:np.ndarray, b: np.ndarray, x_k: np.ndarray, x_ant: np.ndarray, omega: float) -> np.ndarray:
    n_row, n_col = A.shape

    # ascending
    for i in range (n_row):
        l_u = np.sum([A[i][j] * x_k[j] for j in range(n_col) if i != j])
        x_k[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]

    # backward
    for i in range (n_row - 1, -1, -1):
        l_u = np.sum([A[i][j] * x_k[j] for j in range(n_col) if i != j])
        x_k[i] = omega / A[i][i] * (b[i] - l_u) + (1 - omega) * x_k[i]

    return x_k


# methods mappings
methods_map = {
        'Jacobi': {'func': jor_solution, 'name': 'Jacobi'},
        'JOR': {'func': jor_solution, 'name': 'Jacobi Over-Relaxation (JOR)'},
        'GSa': {'func': sora_solution, 'name': 'Gauss–Seidel Ascending (GSa)'},
        'GSb': {'func': sorb_solution, 'name': 'Gauss–Seidel Backward (GSb)'},
        'SORa': {'func': sora_solution, 'name': 'Successive Over-Relaxation Ascending (SORa)'},
        'SORb': {'func': sorb_solution, 'name': 'Successive Over-Relaxation Backward (SORb)'},
        'SGS': {'func': ssor_solution, 'name': 'Symmetric Gauss–Seidel (SGS)'},
        'SSOR': {'func': ssor_solution, 'name': 'Symmetric Successive Over-Relaxation (SSOR)'}
    }


def iterative_solver(A: np.ndarray, b: np.ndarray, x_0: np.ndarray, method: str, omega: float, opt: int, itmax: int = ITMAX, tol: float = TOL) -> dict:
    solver_method = methods_map[method]['func']

    k_f = 0
    converged = False
    errors = []

    x_k = x_0.copy()
    x_ant = x_0.copy()

    while True:
        # compute x_k
        x_k = solver_method(A, b, x_k, x_ant, omega)

        # compute correction/residual
        r_k = np.linalg.norm(b - A @ x_k)
        d_k = np.linalg.norm(x_k - x_ant)

        # choose threshold criterion
        if opt == 0:
            chosen_err_crit = d_k 
        else:
            chosen_err_crit = r_k 

        errors.append((np.round(d_k, 4), np.round(r_k, 4)))

        # check threshold condition
        if chosen_err_crit < tol or k_f >= itmax:
            converged = True
            break

        k_f += 1
        x_ant = x_k.copy()

    result = {'x': x_k, 'k_f': k_f, 'converged': converged, 'errors': errors, 'omega': omega, 'opt' : opt}

    return result


def plot_errors(result: dict, method: str, num_test: int, test: bool = True, save: bool = False):
    errors = result['errors']
    omega = result['omega']
    opt = result['opt']
    iterations = range(len(errors))

    method_name = methods_map[method]['name']

    d_arr = [error[0] for error in errors]
    r_arr = [error[1] for error in errors]

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

    title = f"Test {num_test} for the {method_name} method, " + r"$\omega$" + f" = {omega:.4f}, "
    if opt == 0:
        title += 'stop criterion: correction norm' 
    else:
        title += 'stop criterion: residual norm' 

    fig.suptitle(title)
    
    if test:
        plt.show()

    if save:
        folder_path = f'./best_results_plots/{method}/'
        os.makedirs(folder_path, exist_ok=True)

        filename = f'{num_test}'
        plt.savefig(folder_path + filename + '.pdf')


def save_best_results():
    opts = [0, 1]

    # generate omegas for tests
    omegas_0_1 = np.random.random(20)
    omegas_0_2 = 2 * np.random.random(20)

    for i, test in enumerate(tests, start = 1):
        A, b = test
        _, n_col = A.shape

        x_0 = np.random.randn(n_col)

        for method in methods_map.keys():
            omegas = []

            if method in ['Jacobi', 'GSa', 'GSb', 'SGS']:
                omegas = [1]
            elif method == 'JOR':
                omegas = omegas_0_1
            else:
                omegas = omegas_0_2

            best_method_result = None
            best_k_f = ITMAX + 1

            for omega in omegas:
                for opt in opts:
                    result = iterative_solver(A, b, x_0, method, omega, opt) 
                    if result['k_f'] < best_k_f:
                        best_method_result = (omega, result)
                        best_k_f = result['k_f']


            if best_method_result is not None:
                omega, result = best_method_result
                plot_errors(result, method, i, test = False, save = True)

    plt.close('all')



def main():
    # uncomment next line to run all tests
    # save_best_results() 

    # demo
    test_num = 3 
    method = 'SSOR'

    omega = 0.8
    opt = 1

    A, b = tests[test_num - 1]

    _, n_col = A.shape
    x_0 = np.random.randn(n_col)

    result = iterative_solver(A, b, x_0, method, omega, opt)

    print(('-' * 10) + f' Test {test_num}, {method} method, omega = {omega} ' + ('-' * 10))
    if result['converged']:
        print(f'Converged in {result['k_f']} iterations.')
    else:
        print('Not converged.')

    plot_errors(result, method, test_num, test = True, save = False)


if __name__ == "__main__":
    main()
