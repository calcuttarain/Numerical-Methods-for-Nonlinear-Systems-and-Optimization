function [A, b, x_0] = generate_data(n, kappa, symmetric)
    lambda_mini = 1.e-5;
    lambda_maxi = kappa * lambda_mini;
    lambdas = lambda_mini + (lambda_maxi - lambda_mini) * rand(1, n);
    lambdas(1) = lambda_mini;
    lambdas(2) = lambda_maxi;


    if symmetric
        [Q, ~] = qr(randn(n, n));
        A = Q * diag(lambdas) * Q';
    else
        [U, ~] = qr(randn(n, n));
        [V, ~] = qr(randn(n, n));
        A = U * diag(lambdas) * V';
    end

    b = rand(n, 1);

    x_0 = rand(n, 1);
end
