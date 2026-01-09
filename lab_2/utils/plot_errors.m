function plot_errors(errors_cell, labels, plot_title)
    figure;
    hold on; 
    
    colors = {'r', 'b', 'g', 'm', 'k'};
    
    for j = 1:length(errors_cell)
        err = errors_cell{j};
        err = err(err > 0); 
        
        k = 1:length(err);
        
        color_idx = mod(j-1, length(colors)) + 1;
        
        plot(k, err, colors{color_idx}, 'LineWidth', 2);
    end
    
    grid on;
    set(gca, 'YScale', 'log'); 

    xlabel('Iteration k');
    ylabel('Residual Error');
    title(plot_title);
    
    legend(labels); 
    hold off;
end
