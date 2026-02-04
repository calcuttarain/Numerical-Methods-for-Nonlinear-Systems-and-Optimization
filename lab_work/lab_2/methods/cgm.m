function [x, i, errors] = cgm(A, b, x, itmax, TOL)
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

        w = A * p;

        alpha = rho_curr / (p' * w);

        x = x + alpha * p;

        r = r - alpha * w;

        rho_ant = rho_curr;
        rho_curr = r' * r;
    end

end
