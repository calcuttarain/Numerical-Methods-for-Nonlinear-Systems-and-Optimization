% 6 
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

    K = sparse(toeplitz([2; -1; zeros(n - 2, 1)]));

    T = (1 / h^2) * K;
    
    I_n = speye(n);
    I_N = speye(N);
    
    A_h = (kron(I_n, T) + kron(T, I_n)) + I_N;

    precs = get_precs(A_h, omega);

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

        % cgm
        [u_cgm, kf_cgm, errors_cgm] = cgm(A_h, f_h, u_h_0, itmax, TOL);

        fprintf("%-100s -> %4d iterations | Error (L_2): %.2e\n", "CGM Convergence", kf_cgm, norm(u_exact_sol - u_cgm, 2));
        labels{end+1} = "CGM"; 
        errors_arr{end+1} = errors_cgm;

        % pcgm
        for j = 1:size(precs, 1)
            P = precs{j, 1}; 
            name_p = precs{j, 2};

            [u_pcgm, kf_pcgm, errors_pcgm] = pcgm(A_h, f_h, u_h_0, P, itmax, TOL);

            fprintf("PCGM with %-90s -> %4d iterations | Error (L_2): %.4e\n", name_p, kf_pcgm, norm(u_exact_sol - u_pcgm, 2));
            labels{end+1} = "PCGM with " + name_p;
            errors_arr{end+1} = errors_pcgm;
        end

        plot_title = sprintf("Convergence results for N = %d, l1 = %d, l2 = %d, m1 = %d, m2 = %d", n, l1, l2, m1, m2);
        % uncomment to see plots
        plot_errors(errors_arr, labels, plot_title);

    end
end


% preconditioners
function precs = get_precs(A, omega)
    D = diag(diag(A));
    L = tril(A, -1);
    M = D + omega * L;

    P_J = sparse(D);
    P_col_norm = sparse(diag(sqrt(sum(A.^2, 1))));
    P_SSOR = sparse(M * (D \ M'));

    precs = {
        P_J,        "Jacobi Preconditioner";
        P_col_norm, "Column Norm Preconditioner";
        P_SSOR,     sprintf('Symmetric Successive Over-Relaxation (w=%.2f) Preconditioner', omega);
    };
end
