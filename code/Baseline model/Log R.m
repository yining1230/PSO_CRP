clear
clc

% load datasig_q;
load data_Pd_all_23-44;

for i=1:size(data,2)
    data_norm(:,i)=(data(:,i)-mean(data(:,i)))/std(data(:,i));
end
data = data_norm;

XTrain = data(1:22, :);
YTrain = data1(1:22,2);
XTest = data(23:44, :);
YTest = data1(23:44,2);

YTrain = YTrain - min(YTrain); 
YTest  = YTest  - min(YTest);


logRegModel = fitclinear(XTrain, YTrain, 'Learner', 'logistic', 'Regularization', 'ridge', 'Solver', 'lbfgs');

[~, scores_train] = predict(logRegModel, XTrain); 
[~, scores_test] = predict(logRegModel, XTest);


prob_train = scores_train(:, 2);
prob_test = scores_test(:, 2);


epsilon = 1e-15; 
prob_train = max(min(prob_train, 1 - epsilon), epsilon);
prob_test = max(min(prob_test, 1 - epsilon), epsilon);

cross_entropy_train = -mean(YTrain .* log(prob_train) + (1 - YTrain) .* log(1 - prob_train));
cross_entropy_test = -mean(YTest .* log(prob_test) + (1 - YTest) .* log(1 - prob_test));


fprintf('Training set cross-entropy loss: %.4f\n', cross_entropy_train);
fprintf('Test set cross-entropy loss: %.4f\n', cross_entropy_test);

YPred_train = prob_train >= 0.5;
YPred_test = prob_test >= 0.5;

accuracy_train = sum(YPred_train == YTrain) / length(YTrain);
accuracy_test = sum(YPred_test == YTest) / length(YTest);

fprintf('Training accuracy: %.2f%%\n', accuracy_train * 100);
fprintf('Test accuracy: %.2f%%\n', accuracy_test * 100);
