function [x, i, errors] = sdm(A, b, x, itmax, TOL)
    errors = zeros(1, itmax);

    r = b - A * x;

    norm_b = norm(b, 2);

    for i = 1:itmax
        norm_r = norm(r, 2);

        if norm_r <= TOL * norm_b
            break;
        end
        errors(i) = norm_r;

        w = A * r;

        alpha = norm_r^2 / (r' * w);

        x = x + alpha * r;

        r = r - alpha * w;
    end

end
