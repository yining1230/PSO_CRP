%% Define final prediction function (entire dataset)
X_all = [X_train_top5; X_test_top5];
y_all = [Y_trainresponse; y_test];

% predict_final = @(X) predict(reGP, X(:,1:length(top5Features))) + predict(lm_residuals, X(:,1:length(top5Features)));
predict_final = @(X) predict(regressionGP_top5, X(:,1:length(top5Features))) + predict(lm_residuals, X(:,1:length(top5Features))); % Use this for 1075 and 37 datasets

%% Permutation Feature Importance (PFI) on the entire dataset
num_features = size(X_all, 2);
perm_importance = zeros(num_features, 1);

% Baseline R²
y_pred_base = predict_final(X_all);
TSS_base = sum((y_all - mean(y_all)).^2);
RSS_base = sum((y_all - y_pred_base).^2);
R2_base = 1 - RSS_base / TSS_base;

disp(['Baseline R² on entire dataset: ', num2str(R2_base)]);

for i = 1:num_features
    % Copy the dataset
    X_perm = X_all;
    
    % Shuffle the i-th feature
    X_perm(:, i) = X_perm(randperm(size(X_perm, 1)), i);
    
    % Make predictions with permuted feature
    y_perm = predict_final(X_perm);
    
    % Compute new R²
    RSS_perm = sum((y_all - y_perm).^2);
    R2_perm = 1 - RSS_perm / TSS_base;
    
    % Feature importance = performance drop (ΔR²)
    perm_importance(i) = R2_base - R2_perm;
end

% Sort and display results
[sorted_imp, idx] = sort(perm_importance, 'descend');
disp('Permutation Feature Importance (sorted, entire dataset):');
for j = 1:num_features
    disp(['Feature ', num2str(top5Features(idx(j))), ...
          ' Importance = ', num2str(sorted_imp(j))]);
end

% % Plot all features
% figure;
% bar(sorted_imp);
% set(gca, 'XTickLabel', top5Features(idx));
% xlabel('Feature Index');
% ylabel('Importance (ΔR²)');
% title('Permutation Feature Importance (entire dataset)');
% grid on;

% Select top 10 features
top_n = 10;
top_idx = idx(1:top_n);
top_imp = sorted_imp(1:top_n);

% Display top 10 features
disp(['Top ', num2str(top_n), ' Permutation Feature Importance:']);
for j = 1:top_n
    disp(['Feature ', num2str(top5Features(top_idx(j))), ...
          ' Importance = ', num2str(top_imp(j))]);
end

% Plot bar chart for top 10 features
figure;
bar(top_imp);
set(gca, 'XTickLabel', top5Features(top_idx));
xlabel('Feature Index');
ylabel('Importance (ΔR²)');
title(['Top ', num2str(top_n), ' Permutation Feature Importance']);
grid on;




%% Partial Dependence Plot (PDP) for top 10 features (entire dataset)
top_n = 10;                    
num_grid = 20;                  
top_features_idx = idx(1:top_n); 

figure;
for k = 1:top_n
    feature_idx = top5Features(top_features_idx(k));  
    
    x_values = linspace(min(X_all(:,top_features_idx(k))), max(X_all(:,top_features_idx(k))), num_grid);
    pdp_values = zeros(num_grid,1);

    for j = 1:num_grid
        X_temp = X_all;                                   
        X_temp(:,top_features_idx(k)) = x_values(j);      
        pdp_values(j) = mean(predict_final(X_temp));     
    end
    
    subplot(ceil(top_n/2), 2, k);
    plot(x_values, pdp_values, '-o', 'LineWidth', 1.5);
    xlabel(['Feature ', num2str(feature_idx)]);
    ylabel('Partial Dependence');
    title(['PDP of Feature ', num2str(feature_idx)]);
    grid on;
end


%% Make the image background transparent.
% Delete text objects: title, axis labels, legend
txtObjs = findall(gcf,'Type','text');          % Regular text objects in the figure
axisLabels = findall(gca,'-property','Label'); % XLabel, YLabel
legends = findall(gcf,'Type','Legend');        % Legend objects
delete(txtObjs);
delete(axisLabels);
delete(legends);

% Remove tick labels on axes
set(gca,'XTickLabel',[],'YTickLabel',[]);

% Keep axis lines and ticks visible
set(gca,'XColor','k','YColor','k');  % Black axis lines and ticks

% Set background to transparent
set(gca,'Color','none');
set(gcf,'Color','none');

% Export as EPS
exportgraphics(gca,'plot_axes_only_no_text.eps','ContentType','vector','BackgroundColor','none');