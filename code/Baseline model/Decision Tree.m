%% Decision Tree
clear
clc

% Load data
% load datasig_q;
load data_Pd_all_23-44;

% Normalize the data (z-score standardization)
for i = 1:size(data, 2)
    data_norm(:, i) = (data(:, i) - mean(data(:, i))) / std(data(:, i));
end
data = data_norm;

% Split dataset using indices
XTrain = data(1:22, :);
YTrain = data1(1:22, 2);
XTest = data(23:44, :);
YTest = data1(23:44, 2);

% Train a decision tree classifier
treeModel = fitctree(XTrain, YTrain, ...
    'MaxNumSplits', 1, ...           
    'MinLeafSize', 2, ...             
    'SplitCriterion', 'deviance', ... 
    'CrossVal', 'off');               

YPred_train = predict(treeModel, XTrain);
YPred_test = predict(treeModel, XTest);

accuracy_train = sum(YPred_train == YTrain) / length(YTrain);
accuracy_test = sum(YPred_test == YTest) / length(YTest);

fprintf('Training accuracy: %.2f%%\n', accuracy_train * 100);
fprintf('Test accuracy: %.2f%%\n', accuracy_test * 100);