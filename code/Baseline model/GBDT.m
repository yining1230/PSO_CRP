clear
clc

% load datasig_q;
load data_Pd_all_23-44;


XTrain = data(1:22, :);
YTrain = data1(1:22,2);
XTest = data(23:44, :);
YTest = data1(23:44,2);

gbtModel = fitcensemble(XTrain, YTrain, 'Method', 'AdaBoostM1', 'Learners', 'tree', 'NumLearningCycles', 200);

YPred_train = predict(gbtModel, XTrain);
YPred_test = predict(gbtModel, XTest);

accuracy_train = sum(YPred_train == YTrain) / length(YTrain);
accuracy_test = sum(YPred_test == YTest) / length(YTest);

fprintf('Training accuracy: %.2f%%\n', accuracy_train * 100);
fprintf('Test accuracy: %.2f%%\n', accuracy_test * 100);