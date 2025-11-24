clear
clc

% Read Excel file
rng(1); % Set random seed for reproducibility

% Split dataset using indices
X_trainpredictors = data(:, 1:107);
Y_trainresponse = data(:, 110);
x_test = data1(:, 1:107);
y_test = data1(:, 108);

x = X_trainpredictors;
y = Y_trainresponse;

% Use Particle Swarm Optimization (PSO) to optimize hyperparameters of Gaussian Process Regression
[best_mse, best_ma, best_sigma, best_ell, best_r2, best_r2_cv, best_r2_test] = pso_double_gp(x, y, x_test, y_test);

sigma_f_sq = best_sigma; % Variance parameter
ell = best_ell;          % Length scale parameter
ma = best_ma;            % Noise standard deviation (sigma) for GPR
theta = [sigma_f_sq, ell];
sigma = ma;

% Define custom kernel function
custom_kernel = @(x1, x2, theta) theta(1) * exp(-0.5 * pdist2(x1, x2).^2 / theta(2)^2);

reGP = fitrgp(...
    x, ...
    y, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', custom_kernel, ...
    'KernelParameters', theta, ...
    'Sigma', sigma, ...
    'Standardize', true); % Custom kernel with theta parameters

y_pre = predict(reGP, x);
mse = mean((y - y_pre).^2);
R = corrcoef(y, y_pre);
R_squared = R(1, 2)^2;

%% Perform cross-validation
% partitionedModel = crossval(reGP, 'KFold', 5);
% % Compute validation RMSE
% validationMSE = kfoldLoss(partitionedModel, 'LossFun','mse');
% mse_cv = kfoldLoss(partitionedModel);
% % Compute R²
% var_Y = var(y);
% r2_cv = 1 - mse_cv / var_Y;

%% Compute test set MSE
% Predict on new data
y_testpred = predict(reGP, x_test);
R_test = corrcoef(y_testpred, y_test);
R_testsquared = R_test(1, 2)^2;
mse_test = mean((y_test - y_testpred).^2);
MAE_test = mean(abs(y_test - y_testpred));

%% Plot results
% Plot fitting scatter plot
figure;
scatter(y, y_pre, 'filled');
xlabel('Actual Values');
ylabel('Predicted Values');
% title('Gaussian Process Regression Prediction Scatter Plot');

% Compute slope and intercept of linear fit
coefficients = polyfit(y, y_pre, 1);
slope = coefficients(1);
intercept = coefficients(2);

% Plot fitted line based on slope and intercept
hold on;
plot([min(y), max(y)], slope * [min(y), max(y)] + intercept, 'k-', 'LineWidth', 2);
legend('Training Data Points', 'Fitted Line');
hold on;

% Plot test set predictions
scatter(y_test, y_testpred, 'filled', 'MarkerEdgeColor', [0.5 0.5 0.5]);
legend('Training Data Points', 'Fitted Line', 'Test Set Data Points');
hold on;