clc
clear all;

% Load data
load data_Pd_all_1-22;

trainData = data(23:44, :);
validationData = data(1:22, :);

data = trainData;

% Read external index data from Excel
[data1, textdata1, raw1] = xlsread('\Example_Rxn_Data.xlsx', 'pd_idx', 'A56:C100');
% [data1, textdata1, raw1] = xlsread('C:\Users\DELL\Desktop\hte\rdkit\rdkit_para.xlsx', 'rdall_pidx', 'A1:C1076');
% [data1, textdata1, raw1] = xlsread('C:\Users\DELL\Desktop\aaaa.xlsx', 'sheet8', 'Y1:Z26');
% [data1, textdata1, raw1] = xlsread('C:\Users\29868\Desktop\aaaa.xlsx', 'sheet6', 'A1:B38');

trainData_idx = data1(23:44, 2);
data1 = trainData_idx;

% Normalize training data (z-score standardization)
for i = 1:size(data, 2)
    data_norm(:, i) = (data(:, i) - mean(data(:, i))) / std(data(:, i));
end

% Identify columns containing NaN values
columns_with_nan = any(isnan(data_norm));

% Find column indices that contain NaN
columns_with_nan_indices = find(columns_with_nan);

% % Remove columns containing NaN values
% data_without_nan = data(:, ~columns_with_nan);


% Process validation data
datap = validationData;

% Normalize validation data using its own mean and std (note: usually should use training stats!)
for i = 1:size(datap, 2)
    datap_norm(:, i) = (datap(:, i) - mean(datap(:, i))) / std(datap(:, i));
end

% Identify columns containing NaN values in validation set
columns_with_nan = any(isnan(datap_norm));

% Find column indices that contain NaN
columns_with_nan_indices = find(columns_with_nan);

% % Remove columns containing NaN values
% data_without_nan = data(:, ~columns_with_nan);