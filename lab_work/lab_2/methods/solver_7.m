function [x, idx, errors] = solver_7(h, b, x, method, itmax, TOL, omega)
    N = length(x);
    n = sqrt(N);

    center = 4 / h^2 + 1;
    neighbor = -1 / h^2;

    if nargin < 7
        omega = 1;
    end

    errors = zeros(1, itmax);

    Ax = apply_conv(x, center, neighbor);
    r = b - Ax;
    norm_b = norm(b, 2);
    
    for idx = 1:itmax

        switch method
        case "Conjugate Gradient"
            z = r;

        case "Jacobi Preconditioner"
            z = r / center;

        case "Column Norm Preconditioner"
            val_int = sqrt(center^2 + 4 * neighbor^2); 
            val_edge = sqrt(center^2 + 3 * neighbor^2);
            val_corner = sqrt(center^2 + 2 * neighbor^2);

            val_col_norm = val_int * ones(N, 1);

            % edges
            val_col_norm(1:n) = val_edge; % left
            val_col_norm(N - n + 1 : N) = val_edge; % right
            val_col_norm(1:n:N) = val_edge; % upper
            val_col_norm(n:n:N) = val_edge; % lower

            % corners
            val_col_norm(1) = val_corner;
            val_col_norm(n) = val_corner;
            val_col_norm(N - n + 1) = val_corner;
            val_col_norm(N) = val_corner;

            z = r ./ val_col_norm;

        case "Symmetric Successive Over-Relaxation Preconditioner"
            R = reshape(r, n, n);
    
            % ascending
            Y = zeros(n, n); 
            
            for j = 1:n     
                for i = 1:n  
                    
                    sum_L = 0;
                    
                    if i > 1
                        sum_L = sum_L + neighbor * Y(i-1, j);
                    end
                    
                    if j > 1
                        sum_L = sum_L + neighbor * Y(i, j-1);
                    end
                    
                    Y(i, j) = (R(i, j) - omega * sum_L) / center;
                end
            end
            
            % diagonal scaling
            W = Y * center;
            
            % backward
            Z = zeros(n, n);
            
            for j = n:-1:1 
                for i = n:-1:1 
                    
                    sum_U = 0;
                    
                    if i < n
                        sum_U = sum_U + neighbor * Z(i+1, j);
                    end
                    
                    if j < n
                        sum_U = sum_U + neighbor * Z(i, j+1);
                    end
                    
                    Z(i, j) = (W(i, j) - omega * sum_U) / center;
                end
            end

            z = Z(:);
        end

        rho_curr = r' * z; 
        
        errors(idx) = norm(r, 2);
        if errors(idx) <= TOL * norm_b
            break;
        end
        
        if idx == 1
            p = z;
        else
            beta = rho_curr / rho_ant;
            p = z + beta * p;
        end
        
        Ap = apply_conv(p, center, neighbor);

        alpha = rho_curr / (p' * Ap);
        
        x = x + alpha * p;
        r = r - alpha * Ap;
        
        rho_ant = rho_curr;
    end
end


function x_conv = apply_conv(x, center, neighbor)

    kernel = [0, neighbor, 0; neighbor, center, neighbor; 0, neighbor, 0];

    n = sqrt(length(x));

    % turn to grid
    X_grid = reshape(x, n, n);
    
    % apply convolution
    X_grid_conv = conv2(X_grid, kernel, 'same');
    
    % vectorize back
    x_conv = X_grid_conv(:);
end
