clear; clc;
addpath('utils', 'methods');

rng(10);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TOL = 1.e-15;
itmax = 1000;

kappa = 5000;

n = 1000;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Methods %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SDM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

is_symmetric = true;
[A, b, x_0] = generate_data(n, kappa, is_symmetric);

[~, kf_sdm, errors_sdm] = sdm(A, b, x_0, itmax, TOL);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CGM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P_CGM = eye(n);
[~, kf_cgm, errors_cgm] = pcgm(A, b, x_0, P_CGM, itmax, TOL);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PCGM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = diag(diag(A));
P_J = D;

P_col_norm = diag(sqrt(sum(A.^2, 1)));

omega = 1.1;
L = tril(A, -1);
M = D + omega * L;
P_SSOR = M * (D \ M');

[~, kf_pcgm, errors_pcgm] = pcgm(A, b, x_0, P_SSOR, itmax, TOL);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CGNR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

is_symmetric = false;
[A, b, x_0] = generate_data(n, kappa, is_symmetric);

A = A(1:n - 100, :);
b = b(1:n - 100);

[~, kf_cgnr, errors_cgnr] = cgnr(A, b, x_0, itmax, TOL);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CGNE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~, kf_cgne, errors_cgne] = cgne(A, b, x_0, itmax, TOL);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% symmetric
fprintf("------------------ Sistemul SPD ------------------\n")

fprintf("Convergenta SDM -> %d iteratii.\n", kf_sdm);
fprintf("Convergenta CGM -> %d iteratii.\n", kf_cgm);
fprintf("Convergenta PCGM -> %d iteratii.\n", kf_pcgm);

errors = {errors_sdm, errors_cgm, errors_pcgm};
labels = {"Steepest Descend", "Conjugate Gradient", "Preconditioned Conjugate Gradient"};
plot_title = "Symmetric System Convergence Comparison";

plot_errors(errors, labels, plot_title);

% unsymmetric
fprintf("\n------------------ Sistemul Nesimetric ------------------\n")

fprintf("Convergenta CGNR -> %d iteratii.\n", kf_cgnr);
fprintf("Convergenta CGNE -> %d iteratii.\n", kf_cgne);

errors = {errors_cgnr, errors_cgne};
labels = {"CG Normal Equations Residual", "CG Normal Equations Error"};
plot_title = "Unsymmetric System Convergence Comparison";

plot_errors(errors, labels, plot_title);
