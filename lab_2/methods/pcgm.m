function [x, i, errors] = pcgm(A, b, x, P, itmax, TOL)
    errors = zeros(1, itmax);

    r = b - A * x;
    norm_b = norm(b, 2);
    
    for i = 1:itmax
        z = P \ r; 
        
        rho_curr = r' * z; 
        
        errors(i) = norm(r, 2);
        if errors(i) <= TOL * norm_b
            break;
        end
        
        if i == 1
            p = z;
        else
            beta = rho_curr / rho_ant;
            p = z + beta * p;
        end
        
        w = A * p;
        alpha = rho_curr / (p' * w);
        
        x = x + alpha * p;
        r = r - alpha * w;
        
        rho_ant = rho_curr;
    end
end
