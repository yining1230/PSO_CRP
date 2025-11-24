
x = X_train_top5;
y = Y_trainresponse;
x_test = X_test_top5;
y_test;
% Use the PSO algorithm to optimize the hyperparameters of Gaussian Process Regression.
[best_mse, best_ma, best_sigma, best_ell, best_r2, best_r2_cv, best_r2_test] = pso_double_gp(x, y, x_test, y_test)

sigma_f_sq = best_sigma; % Variance parameter

ell = best_ell; % Length scale parameter
ma = best_ma;% sigma
theta = [sigma_f_sq,ell];
sigma = ma;
% Define a custom kernel function
custom_kernel = @(x1, x2, theta) theta(1) * exp(-0.5 * pdist2(x1, x2).^2 / theta(2)^2);
reGP = fitrgp(...
    x, ...
    y, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', custom_kernel, ...
    'KernelParameters', theta, ...
    'Sigma', sigma,'Standardize', true);

y_pre = predict(reGP, x);
mse = mean((y - y_pre).^2);
yMean = mean(y);
TSS = sum((y - yMean).^2); 
RSS = sum((y - y_pre).^2); 
R_squared = 1 - RSS/TSS;

% the validation set
y_testpred = predict(reGP, x_test);
yMean_test = mean(y_test);
TSS_test = sum((y_test - yMean_test).^2); 
RSS_test = sum((y_test - y_testpred).^2); 
R_testsquared = 1 - RSS_test/TSS_test;

mse_test = mean((y_test - y_testpred).^2);
MAE_test = mean(abs(y_test - y_testpred));

% Calculate residuals
residuals = Y_trainresponse - y_pre;
% MLR
lm_residuals = stepwiselm(X_train_top5, residuals, 'Upper', 'linear', 'Verbose', 2);
% lm_residuals = fitlm(X_trainpredictors, residuals);  
disp(lm_residuals);

residual_pred = predict(lm_residuals, X_train_top5);


final_predictions = y_pre + residual_pred;

disp(final_predictions);

% combine
yMean_train5 = mean(Y_trainresponse);
TSS_train5 = sum((Y_trainresponse - yMean_train5).^2); 
RSS_train5 = sum((Y_trainresponse - final_predictions).^2); 
R2_train5 = 1 - RSS_train5/TSS_train5;

Y_linear_test = predict(lm_residuals, X_test_top5);  
Y_combined_test = y_testpred + Y_linear_test;  

yMean_test5 = mean(y_test);
TSS_test5 = sum((y_test - yMean_test5).^2); 
RSS_test5 = sum((y_test - Y_combined_test).^2); 
R2_test5 = 1 - RSS_test5/TSS_test5;
MAE_test = mean(abs(y_test - Y_combined_test));

disp(['Training R^2 (outcome): ', num2str(R2_train5)]);
disp(['Testing R^2 (outcome): ', num2str(R2_test5)]);
