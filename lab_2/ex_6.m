% 6 
clc; clear;
addpath('utils', 'methods');

TOL = 1.e-15;
itmax = 10000;
omega = 1.9;

n_arr = [49 99 199 399 799];
l_m_arr = [2 2; 10 2; 2 10]';


% define functions
u_ex = @(x, l, m) x.^l .* (1 - x).^m;

u_ex_2_der = @(x, l, m) l * (l - 1) * x.^(l - 2) .* (1 - x).^m ...
                        - 2 * l * m * x.^(l - 1) .* (1 - x).^(m - 1) ...
                        + m * (m - 1) * x.^l .* (1 - x).^(m - 2);

f_x = @(x, l, m) - u_ex_2_der(x, l, m) + u_ex(x, l, m);

for i = 1:length(n_arr)
    n = n_arr(i);
    h = 1 / (n + 1);

    K = toeplitz([2; -1; zeros(n - 2, 1)]);
    A_h = 1 / h^2 * K + eye(n);

    precs = get_precs(A_h, omega);

    x = h * (1:n)';

    fprintf("\n-------------------------------------- N = %d --------------------------------------\n", n);

    for pair = l_m_arr 
        labels = {};
        errors_arr = {};

        l = pair(1);
        m = pair(2);

        fprintf("\n------------- l = %d, m = %d -------------\n", l, m);


        u_h_0 = rand(n, 1);
        f_h = f_x(x, l, m);

        u_exact_sol = u_ex(x, l, m);

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

        plot_title = sprintf("Convergence results for N = %d, l = %d, m = %d", n, l, m);
        % uncomment to see plots
        % plot_errors(errors_arr, labels, plot_title);

    end
end


% preconditioners
function precs = get_precs(A, omega)
    D = diag(diag(A));
    L = tril(A, -1);
    M = D + omega * L;

    P_J = D;
    P_col_norm = diag(sqrt(sum(A.^2, 1)));
    P_SSOR = M * (D \ M');

    precs = {
        P_J,        "Jacobi Preconditioner";
        P_col_norm, "Column Norm Preconditioner";
        P_SSOR,     sprintf('Symmetric Successive Over-Relaxation (w=%.2f) Preconditioner', omega);
    };
end
