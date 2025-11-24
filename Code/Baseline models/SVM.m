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

svmModel = fitcsvm(XTrain, YTrain, ...
    'KernelFunction', 'rbf', ...        
    'BoxConstraint', 1)           


YPred_test = predict(svmModel, XTest);
YPred_train = predict(svmModel, XTrain);


accuracy_test = sum(YPred_test == YTest) / length(YTest);
accuracy_train = sum(YPred_train == YTrain) / length(YTrain);

fprintf('Training accuracy: %.2f%%\n', accuracy_train * 100);
fprintf('Test accuracy: %.2f%%\n', accuracy_test * 100);
