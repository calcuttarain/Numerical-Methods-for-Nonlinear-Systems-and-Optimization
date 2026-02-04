function [x, i, errors] = cgne(A, b, x, itmax, TOL)
    errors = zeros(1, itmax);

    r = b - A * x;

    rho_curr = r' * r; rho_ant = 0;

    norm_b = norm(b, 2);

    for i = 1:itmax

        if sqrt(rho_curr) <= TOL * norm_b
            break;
        end
        errors(i) = sqrt(rho_curr);

        if i == 1
            p = r;
        else 
            beta_k = rho_curr / rho_ant;
            p = r + beta_k * p;
        end

        w = A' * p;

        alpha = rho_curr / (w' * w);

        x = x + alpha * w;

        r = r - alpha * (A * w);

        rho_ant = rho_curr;
        rho_curr = r' * r;
    end

end

