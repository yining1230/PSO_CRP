clear
clc

% Read Excel file
[data, txt, raw] = xlsread('E:\PhD_Year2\anli\hte\rdkit\rdkit_para.xlsx', 'rdkit', 'A1:XE1076');

% Extract parameter names from the first row
parameters = raw(1, :); % The first row contains parameter names

% Modify parameter names
for i = 2:628
    if i >= 2 && i <= 210
        parameters{i} = ['cat_' parameters{i}];
    elseif i >= 211 && i <= 419
        parameters{i} = ['imine_' parameters{i}];
    elseif i >= 420 && i <= 628
        parameters{i} = ['thiol_' parameters{i}];
    end
end

% Replace the modified parameter names back into the original table
raw(1, :) = parameters;

% Save the modified Excel file
outputFileName = 'rdkit_para_modified.xlsx';
xlswrite(outputFileName, raw);

disp(['Modified file has been saved as: ', outputFileName]);