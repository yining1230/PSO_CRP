clear
clc

% load datasig_q;
load data_Pd_all_23-44;

XTrain = data(1:22, :)';
YTrain = data1(1:22,2)';
XTest = data(23:44, :)';
YTest = data1(23:44,2)';

YTrain_categorical = full(ind2vec(YTrain));  
YTest_categorical = full(ind2vec(YTest));    


hiddenLayerSize = 5;  
net = patternnet(hiddenLayerSize);
net.performFcn = 'crossentropy';  


net.trainParam.epochs = 300;  
net.trainParam.showWindow = false; 


[net, tr] = train(net, XTrain, YTrain_categorical);

YPred_train = net(XTrain);
YPred_test = net(XTest);

[~, YPred_train_labels] = max(YPred_train);
[~, YPred_test_labels] = max(YPred_test);

accuracy_train = sum(YPred_train_labels == YTrain) / length(YTrain);
accuracy_test = sum(YPred_test_labels == YTest) / length(YTest);


fprintf('Training accuracy: %.2f%%\n', accuracy_train * 100);
fprintf('Test accuracy: %.2f%%\n', accuracy_test * 100);