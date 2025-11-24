clear
clc
[data,textdata,raw]=xlsread('E:\博二\anli\sigman_37\之前\0915rdkitqm1112.xlsx','rdkitall','A1:DE38');
rng(1);% seed

% dataset split
trainRatio = 0.7;
testRatio = 0.3;  

totalSize = size(data, 1);

shuffledIndices = randperm(totalSize);

trainSize = round(totalSize * trainRatio);
testSize = totalSize - trainSize; 

trainIndices = shuffledIndices(1:trainSize);
testIndices = shuffledIndices(trainSize+1:end);

X_train = data(trainIndices, 1:107); 
Y_train = data(trainIndices, 108);   
X_test = data(testIndices, 1:107);   
Y_test = data(testIndices, 108);     

X_trainpredictors = X_train;
Y_trainresponse = Y_train;
x_test = X_test;
y_test = Y_test;

% set a Gaussian process regression model
sigma = 2;

regressionGP_zs = fitrgp(...
    X_trainpredictors, ...
    Y_trainresponse, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'exponential', ...
    'Sigma', sigma, 'Standardize', 1);

Y_trainpre = predict(regressionGP_zs, X_trainpredictors);

yMean_train = mean(Y_trainresponse);
TSS_train = sum((Y_trainresponse - yMean_train).^2); % 总平方和
RSS_train = sum((Y_trainresponse - Y_trainpre).^2); % 残差平方和
R2_train = 1 - RSS_train/TSS_train;

[y_pred,y_sd] = predict(regressionGP_zs, x_test);

yMean_test = mean(y_test);
TSS_test = sum((y_test - yMean_test).^2); % 总平方和
RSS_test = sum((y_test - y_pred).^2); % 残差平方和
R2_test = 1 - RSS_test/TSS_test;


% 4-fold Cross-validation
cvRegressionGP = crossval(regressionGP_zs, 'KFold', 4);

mse = kfoldLoss(cvRegressionGP);

yFit = kfoldPredict(cvRegressionGP);

yMean_cv = mean(Y_trainresponse);
TSS_cv = sum((Y_trainresponse - yMean_cv).^2);
RSS_cv = sum((Y_trainresponse - yFit).^2);
R2_cv = 1 - RSS_cv/TSS_cv;

% output
disp(['R²: ', num2str(R2_train)]);
disp(['the R² of 4-fold cv: ', num2str(R2_cv)]);
disp(['the R² of the test: ', num2str(R2_test)]);

%% the feature importance
k = 4;
cv = cvpartition(size(X_trainpredictors, 1), 'KFold', k);

num_features = size(X_trainpredictors, 2);
feature_importance = zeros(num_features, 1);

R2_train_original = zeros(k, 1); % 用来存储每个折叠的R²

for i = 1:k
    idx_train = training(cv, i);
    idx_val = test(cv, i);
    X_train = X_trainpredictors(idx_train, :);
    Y_train = Y_trainresponse(idx_train);
    X_val = X_trainpredictors(idx_val, :);
    Y_val = Y_trainresponse(idx_val);
    
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
        Y_train = Y_trainresponse(idx_train);
        X_val = X_trainpredictors(idx_val, :);
        Y_val = Y_trainresponse(idx_val);
        
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

disp('Feature importance (ranked in descending order)：');
[sorted_importance, idx] = sort(feature_importance, 'descend');
for i = 1:num_features
    disp(['Feature', num2str(idx(i)), ': ', num2str(sorted_importance(i))]);
end

% Select the top X features.
top5Features = idx(1:76);

X_train_top5 = X_trainpredictors(:, top5Features);
X_test_top5 = x_test(:, top5Features);

regressionGP_top5 = fitrgp(...
    X_train_top5, ...
    Y_trainresponse, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'exponential', ...
    'Sigma', sigma, 'Standardize', 1);
%     'KernelFunction', 'exponential', ...


% train set
Y_trainpre_top5 = predict(regressionGP_top5, X_train_top5);
yMean_train5 = mean(Y_trainresponse);
TSS_train5 = sum((Y_trainresponse - yMean_train5).^2); % 总平方和
RSS_train5 = sum((Y_trainresponse - Y_trainpre_top5).^2); % 残差平方和
R2_train5 = 1 - RSS_train5/TSS_train5;

% test set
[y_pred_top5, y_sd_top5] = predict(regressionGP_top5, X_test_top5);
yMean_test5 = mean(y_test);
TSS_test5 = sum((y_test - yMean_test5).^2); % 总平方和
RSS_test5 = sum((y_test - y_pred_top5).^2); % 残差平方和
R2_test5 = 1 - RSS_test5/TSS_test5;

disp(['Training R^2 (Top 5 Features): ', num2str(R2_train5)]);
disp(['Testing R^2 (Top 5 Features): ', num2str(R2_test5)]);

% Calculate residuals
residuals = Y_trainresponse - Y_trainpre_top5;
% MLR
lm_residuals = stepwiselm(X_train_top5, residuals, 'Upper', 'linear', 'Verbose', 2);
% lm_residuals = fitlm(X_trainpredictors, residuals); 
disp(lm_residuals);


residual_pred = predict(lm_residuals, X_train_top5);

final_predictions = Y_trainpre_top5 + residual_pred;

disp(final_predictions);

% combine
yMean_train5 = mean(Y_trainresponse);
TSS_train5 = sum((Y_trainresponse - yMean_train5).^2);
RSS_train5 = sum((Y_trainresponse - final_predictions).^2); 
R2_train5 = 1 - RSS_train5/TSS_train5;

Y_linear_test = predict(lm_residuals, X_test_top5);  
Y_combined_test = y_pred_top5 + Y_linear_test;  

yMean_test5 = mean(y_test);
TSS_test5 = sum((y_test - yMean_test5).^2); 
RSS_test5 = sum((y_test - Y_combined_test).^2); 
R2_test5 = 1 - RSS_test5/TSS_test5;
MAE_test = mean(abs(y_test - Y_combined_test));

disp(['Training R^2 (outcome): ', num2str(R2_train5)]);
disp(['Testing R^2 (outcome): ', num2str(R2_test5)]);

MAE_test = mean(abs(y_test - Y_combined_test));
MAE_train = mean(abs(Y_trainresponse - final_predictions));

disp(['Training R^2 (outcome): ', num2str(R2_train5)]);
disp(['Training MAE (outcome): ', num2str(MAE_train)]);
disp(['Testing R^2 (outcome): ', num2str(R2_test5)]);
disp(['Testing MAE (outcome): ', num2str(MAE_test)]);


RMSE_train = sqrt(mean((Y_trainresponse - final_predictions).^2));

RMSE_test = sqrt(mean((y_test - Y_combined_test).^2));

disp(['Train RMSE: ', num2str(RMSE_train)]);
disp(['Testing RMSE: ', num2str(RMSE_test)]);

