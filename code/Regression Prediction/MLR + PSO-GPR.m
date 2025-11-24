clear
clc

rng(12);

totalSize = size(data1, 1); 

shuffledIndices = randperm(totalSize);


trainSize = 600; 
testSize = 475;
trainIndices = shuffledIndices(1:trainSize);
testIndices = shuffledIndices(trainSize+1:trainSize+testSize);

X_train = data1(trainIndices, 1:627); 
Y_train = data1(trainIndices, end); 
X_test = data1(testIndices, 1:627); 
Y_test = data1(testIndices, end); 

X_trainpredictors = X_train;
Y_trainresponse = Y_train;
X_test = X_test;
Y_test = Y_test;


[X_train_normalized, mu, sigma] = zscore(X_trainpredictors);
[X_test_normalized, mu, sigma] = zscore(X_test);

lm =stepwiselm(X_train_normalized, Y_trainresponse, 'Upper', 'linear', 'Verbose', 2);
disp(lm);

% The residual model PSO_gpr
Y_trainlinear = predict(lm, X_train_normalized);  
residuals_train = Y_trainresponse - Y_trainlinear;

Y_testlinear = predict(lm, X_test_normalized);  
residuals_test = Y_test - Y_testlinear; 

% GPR
sigma = 2;

regressionGP_zs = fitrgp(...
    X_trainpredictors, ...
    residuals_train, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'exponential', ...
    'Sigma', sigma, 'Standardize', 1);

Y_trainresiduals_pre = predict(regressionGP_zs, X_trainpredictors);

yMean_train = mean(residuals_train);
TSS_train = sum((residuals_train - yMean_train).^2);
RSS_train = sum((residuals_train - Y_trainresiduals_pre).^2); 
R2_train = 1 - RSS_train/TSS_train;

[Y_trainresiduals_pred,Y_sd] = predict(regressionGP_zs, X_test);

yMean_test = mean(residuals_test);
TSS_test = sum((residuals_test - yMean_test).^2); 
RSS_test = sum((residuals_test - Y_trainresiduals_pred).^2); 
R2_test = 1 - RSS_test/TSS_test;


% 4-fold CV
cvRegressionGP = crossval(regressionGP_zs, 'KFold', 4);

% MSE
mse = kfoldLoss(cvRegressionGP);

yFit = kfoldPredict(cvRegressionGP);
yMean_cv = mean(residuals_train);
TSS_cv = sum((residuals_train - yMean_cv).^2);
RSS_cv = sum((residuals_train - yFit).^2);

R2_cv = 1 - RSS_cv/TSS_cv;

disp(['R² on the training set: ', num2str(R2_train)]);
disp(['R² from 4-fold cross-validation: ', num2str(R2_cv)]);
disp(['R² on the testing set: ', num2str(R2_test)]);

%% Compute feature importance.

k = 4;
cv = cvpartition(size(X_trainpredictors, 1), 'KFold', k);

% Initialize feature importance.

num_features = size(X_trainpredictors, 2);
feature_importance = zeros(num_features, 1);

R2_train_original = zeros(k, 1);

for i = 1:k
    idx_train = training(cv, i);
    idx_val = test(cv, i);
    X_train = X_trainpredictors(idx_train, :);
    Y_train = residuals_train(idx_train);
    X_val = X_trainpredictors(idx_val, :);
    Y_val = residuals_train(idx_val);
    
    % GPR model
    regressionGP = fitrgp(X_train, Y_train, 'BasisFunction', 'constant', 'KernelFunction', 'exponential', 'Sigma', sigma, 'Standardize', 1);
    
    Y_pred = predict(regressionGP, X_val);
    
    yMean = mean(Y_val);
    TSS = sum((Y_val - yMean).^2); 
    RSS = sum((Y_val - Y_pred).^2); 
    R2_train_original(i) = 1 - RSS / TSS;
end


R2_train_original_mean = mean(R2_train_original);

% Remove features one by one and perform cross-validation.
for i = 1:num_features
    R2_train_reduced = zeros(k, 1); 

    for j = 1:k
        idx_train = training(cv, j);
        idx_val = test(cv, j);
        X_train = X_trainpredictors(idx_train, :);
        Y_train = residuals_train(idx_train);
        X_val = X_trainpredictors(idx_val, :);
        Y_val = residuals_train(idx_val);
        
        % Remove the i-th feature.
        X_train_reduced = X_train(:, setdiff(1:num_features, i));
        X_val_reduced = X_val(:, setdiff(1:num_features, i));
        
        regressionGP_reduced = fitrgp(X_train_reduced, Y_train, 'BasisFunction', 'constant', 'KernelFunction', 'exponential', 'Sigma', sigma, 'Standardize', 1);
        
        Y_pred_reduced = predict(regressionGP_reduced, X_val_reduced);
        
        yMean_reduced = mean(Y_val);
        TSS_reduced = sum((Y_val - yMean_reduced).^2); 
        RSS_reduced = sum((Y_val - Y_pred_reduced).^2);
        R2_train_reduced(j) = 1 - RSS_reduced / TSS_reduced;
    end

    R2_train_reduced_mean = mean(R2_train_reduced);
    
    feature_importance(i) = R2_train_original_mean - R2_train_reduced_mean;
end

% Output feature importance.

disp('Feature importance (sorted in descending order)：');
[sorted_importance, idx] = sort(feature_importance, 'descend');
for i = 1:num_features
    disp(['Feature ', num2str(idx(i)), ': ', num2str(sorted_importance(i))]);
end

% Select the top X features.
top5Features = idx(1:73);


X_train_top5 = X_trainpredictors(:, top5Features);
X_test_top5 = X_test(:, top5Features);

x = X_train_top5;
y = residuals_train;
x_test = X_test_top5;
y_test = residuals_test;

% Invoke the PSO algorithm to optimize the hyperparameters of GPR.

[best_mse, best_ma, best_sigma, best_ell, best_r2, best_r2_cv, best_r2_test] = pso_double_gp(x, y, x_test, y_test)

sigma_f_sq = best_sigma; 
ell = best_ell; 
ma = best_ma;
theta = [sigma_f_sq,ell];
sigma = ma;

% Define a custom kernel function.
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
final_predictions = y_pre + Y_trainlinear;


disp(final_predictions);

% combine
yMean_train5 = mean(Y_trainresponse);
TSS_train5 = sum((Y_trainresponse - yMean_train5).^2);
RSS_train5 = sum((Y_trainresponse - final_predictions).^2); 
R2_train5 = 1 - RSS_train5/TSS_train5;
MAE_train = mean(abs(Y_trainresponse - final_predictions));


y_testpred = predict(reGP, x_test);
yMean_test = mean(y_test);
TSS_test = sum((y_test - yMean_test).^2); 
RSS_test = sum((y_test - y_testpred).^2); 
R_testsquared = 1 - RSS_test/TSS_test;

% mse_test = mean((y_test - y_testpred).^2);
% MAE_test = mean(abs(y_test - y_testpred));


% test set
Y_combined_test = y_testpred + Y_testlinear;  

yMean_test5 = mean(Y_test);
TSS_test5 = sum((Y_test - yMean_test5).^2);
RSS_test5 = sum((Y_test - Y_combined_test).^2); 
R2_test5 = 1 - RSS_test5/TSS_test5;
MAE_test = mean(abs(Y_test - Y_combined_test));

disp(['Training R^2 (outcome): ', num2str(R2_train5)]);
disp(['Training MAE (outcome): ', num2str(MAE_train)]);
disp(['Testing R^2 (outcome): ', num2str(R2_test5)]);
disp(['Testing MAE (outcome): ', num2str(MAE_test)]);


RMSE_train = sqrt(mean((Y_trainresponse - final_predictions).^2));

RMSE_test = sqrt(mean((Y_test - Y_combined_test).^2));


disp(['Train RMSE: ', num2str(RMSE_train)]);
disp(['Testing RMSE: ', num2str(RMSE_test)]);

