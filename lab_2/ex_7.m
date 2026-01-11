% 6 

% Observatie: matricile A, P_j si P_column_norm nu trebuie stocate in memorie. 
% In loc de inmultirea matricii A cu vector, se poate folosi convolutie pe gridul U, intrucat Laplacianul e media vecinilor.
clc; clear;
addpath('utils', 'methods');

TOL = 1.e-15;
itmax = 10000;
omega = 1.9;

n_arr = [49 99 199 399];
l_m_arr = [
    2  2  2  2;   
    10 10 2  2;  
    10 2  10 2; 
    2  10 2  10;
    2  2  10 10
]';

methods = ["Conjugate Gradient", "Jacobi Preconditioner", "Column Norm Preconditioner", "Symmetric Successive Over-Relaxation Preconditioner"];


% define functions
g = @(t, l, m) t.^l .* (1 - t).^m;

g_dd = @(t, l, m) l * (l - 1) * t.^(l - 2) .* (1 - t).^m ...
                  - 2 * l * m * t.^(l - 1) .* (1 - t).^(m - 1) ...
                  + m * (m - 1) * t.^l .* (1 - t).^(m - 2);

u_ex_2d = @(x, y, l1, l2, m1, m2) g(x, l1, m1) .* g(y, l2, m2);

delta_u = @(x, y, l1, l2, m1, m2) g_dd(x, l1, m1) .* g(y, l2, m2) ...
                                + g(x, l1, m1) .* g_dd(y, l2, m2);

f_x = @(x, y, l1, l2, m1, m2) -delta_u(x, y, l1, l2, m1, m2) + u_ex_2d(x, y, l1, l2, m1, m2);


for i = 1:length(n_arr)
    n = n_arr(i);
    N = n^2;
    h = 1 / (n + 1);

    v = h * (1:n);
    [X, Y] = meshgrid(v, v);
    
    x_vec = X(:);
    y_vec = Y(:);

    x = h * (1:n)';

    fprintf("\n====================================== Grid N = %d x %d (Total: %d) ======================================\n", n, n, N);

    for col_idx = 1:size(l_m_arr, 2)
        params = l_m_arr(:, col_idx);
        l1 = params(1); l2 = params(2);
        m1 = params(3); m2 = params(4);
        labels = {};
        errors_arr = {};

        fprintf("\n--- l1 = %d, l2 = %d, m1 = %d, m2 = %d ---\n", l1, l2, m1, m2);

        u_h_0 = zeros(N, 1);
        f_h = f_x(x_vec, y_vec, l1, l2, m1, m2);

        u_exact_sol = u_ex_2d(x_vec, y_vec, l1, l2, m1, m2);

        for j = 1:length(methods)
            method = methods(j);

            [u, kf, errors] = solver_7(h, f_h, u_h_0, method, itmax, TOL, omega);

            fprintf("%-90s -> %4d iterations | Error (L_2): %.4e\n", method, kf, norm(u_exact_sol - u, 2));

            errors_arr{end+1} = errors;
        end

        plot_title = sprintf("Convergence results for N = %d, l1 = %d, l2 = %d, m1 = %d, m2 = %d", n, l1, l2, m1, m2);
        % uncomment to see plots
        % plot_errors(errors_arr, methods, plot_title);

    end
end
