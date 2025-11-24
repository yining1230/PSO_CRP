% Define the Gaussian Process Regression function
function [fitness, fitnessma, fitnesssigma, fitnessell,fitness_r2, fitness_r2_cv, fitness_r2_test] = gp_double_regression(x, y, x_test, y_test, para)

    sigma_f_sq = para(1);
    ell = para(2);
    ma = para(3);
    theta = [sigma_f_sq,ell];
    sigma = ma;
    % Compute the covariance matrix
    N = size(x, 1);
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
%     R = corrcoef(y, y_pre);
%     R_squared = R(1, 2)^2; 
    yMean = mean(y);
    TSS = sum((y - yMean).^2); 
    RSS = sum((y - y_pre).^2); 
    R_squared = 1 - RSS/TSS;

    %% CV
    partitionedModel = crossval(reGP, 'KFold', 5);
    validationMSE = kfoldLoss(partitionedModel, 'LossFun','mse');
    mse_cv = kfoldLoss(partitionedModel);
    var_Y = var(y);
    r2_cv = 1 - mse_cv / var_Y;

    y_testpred = predict(reGP, x_test);
%     R_test = corrcoef(y_testpred, y_test);
%     R_testsquared = R_test(1, 2)^2;
    mse_test = mean((y_test - y_testpred).^2);

    yMean_test = mean(y_test);
    TSS_test = sum((y_test - yMean_test).^2); 
    RSS_test = sum((y_test - y_testpred).^2); 
    R_testsquared = 1 - RSS_test/TSS_test;

    fitness = mse;
    fitnessma = sigma;
    fitnesssigma = sigma_f_sq;
    fitnessell = ell;
    fitness_r2 = R_squared;
    fitness_r2_cv = r2_cv;
    fitness_r2_test = R_testsquared;
    
end