function [x, i, errors] = cgnr(A, b, x, itmax, TOL)
    errors = zeros(1, itmax);

    r = b - A * x;
    s = A' * r;
    rho_curr = s' * s; rho_ant = 0;

    norm_b = norm(b, 2);

    for i = 1:itmax

        if norm(r, 2) <= TOL * norm_b
            break;
        end
        errors(i) = norm(r, 2);

        if i == 1
            p = A' * r;
        else 
            beta_k = rho_curr / rho_ant;
            p = s + beta_k * p;
        end

        q = A * p;

        alpha = rho_curr / (q' * q);

        x = x + alpha * p;

        r = r - alpha * q;

        s = A' * r;

        rho_ant = rho_curr;
        rho_curr = s' * s;
    end

end
