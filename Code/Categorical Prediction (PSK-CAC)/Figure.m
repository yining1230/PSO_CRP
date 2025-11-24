clear all;
clc;

% Load data
load 170_data_170_1019pre.mat

%% Data Preparation
X = X_kpca;
X_pre = datap_kpca;
X_all = [X; X_pre];
% shuffledIndices = (1:44).'
X_nameall = [shuffledIndices, X_all];
idx_train = fitnessclugbesthuizong;
idx_test = idx_pre;
idx_all = [idx_train; idx_test];
labels = cellstr(num2str(X_nameall(:, 1))); % Convert numeric IDs to string format

% Extract parameter columns (excluding the ID column)
X_params = X_nameall(:, 2:end);

%% Construct Global Distance Matrix
n_points = size(X_params, 1);
dist_matrix = inf(n_points); % Initialize distance matrix with infinity

% Compute intra-cluster distances
unique_clusters = unique(idx_all);
for i = 1:length(unique_clusters)
    cluster_idx = find(idx_all == unique_clusters(i));
    if length(cluster_idx) > 1
        intra_distances = pdist(X_params(cluster_idx, :), 'euclidean');
        intra_square = squareform(intra_distances); % Convert to square matrix
        dist_matrix(cluster_idx, cluster_idx) = intra_square;
    end
end

% **Artificially increase inter-cluster distances**
% Iterate over all pairs of clusters
for i = 1:length(unique_clusters)
    for j = i+1:length(unique_clusters)
        cluster_idx1 = find(idx_all == unique_clusters(i));
        cluster_idx2 = find(idx_all == unique_clusters(j));
        % Set a large artificial inter-cluster distance (e.g., 2)
        dist_matrix(cluster_idx1, cluster_idx2) = 2;
        dist_matrix(cluster_idx2, cluster_idx1) = 2;
    end
end

% Set diagonal to zero
dist_matrix(1:n_points+1:end) = 0;

% Convert full distance matrix to vector form (as required by linkage)
dist_vector = squareform(dist_matrix);

%% Generate Dendrogram Using Linkage
Z = linkage(dist_vector, 'average');

XX = X_kpca;
X_pre = datap_kpca;
X_all = [XX; X_pre];
X_nameall = [shuffledIndices, X_all];
X = X_all';
Y = X;
Data = abs(corr(X));
labels = cellstr(num2str(X_nameall(:, 1)));
colName = labels;
rowName = colName;
[rows, cols] = size(Data);
fig = figure('Position', [500, 200, 800, 750], 'Name', 'Hierarchical Clustering Plot', 'Color', [1, 1, 1]);

% Create axes layout =============================================================
% Main heatmap axes
placeMat = zeros(7, 8);
placeMat(2:7, 2:7) = 1;
axMain = subplot(7, 8, find(placeMat'));
axMain.XLim = [1, cols] + [-.5, .5];
axMain.YLim = [1, rows] + [-.5, .5];
axMain.YAxisLocation = 'right';
axMain.YDir = 'reverse';
axMain.XTick = 1:cols;
axMain.YTick = 1:rows;
axMain.XTickLabelRotation = 0;

% axMain.FontName = 'Times New Roman'; % Set font to Times New Roman
axMain.FontName = 'Arial';   % Set font to Arial

axMain.FontSize = 7.5; % Set font size
axMain.FontWeight = 'bold'; % Bold font

hold on

% Left dendrogram axes
axTree1 = subplot(7, 8, (1:6).*8 + 1);
axTree1.Position(3) = axTree1.Position(3) + axTree1.Position(1)*4/5;
axTree1.Position(1) = axTree1.Position(1)/5;
axTree1.Position(3) = (axMain.Position(1) - axTree1.Position(1) - axTree1.Position(3))*4/5 + axTree1.Position(3);
hold on

% Top dendrogram axes
axTree2 = subplot(7, 8, 2:7);
axTree2.Position(2) = axMain.Position(2) + axMain.Position(4) + (axTree2.Position(2) - axMain.Position(2) - axMain.Position(4))/5;
axTree2.Position(4) = axTree2.Position(4) + (1 - axTree2.Position(2) - axTree2.Position(4))*4/5;
hold on

% Colorbar axes
axBar = subplot(7, 8, (1:7).*8);
axBar.Position(1) = 1/2;
axBar.Position(3) = .92/2;
axBar.Position(2) = axMain.Position(2) + axMain.Position(4)/2;
axBar.Position(4) = axMain.Position(2) + axMain.Position(4) - axBar.Position(2);
axBar.Color = 'none';
axBar.XColor = 'none';
axBar.YColor = 'none';
uistack(axBar, 'bottom')

CM = colorbar;

% Adjust colorbar position
CM.Position(1) = CM.Position(1) - 0.05; % Shift left
CM.Position(2) = CM.Position(2) - 0.15; % Shift down

% Adjust colorbar font and ticks
% CM.Label.String = 'Correlation Heatmap Scale';    % Colorbar title
CM.Label.Rotation = 90;             % Vertical label
CM.Label.Position = [-1, 0];        % Label position
CM.Label.FontSize = 10;             % Title font size
CM.FontSize = 8;                    % Tick label font size
% CM.FontName = 'Times New Roman';  % Font style

CM.FontName = 'Arial';              % Set font to Arial
CM.FontWeight = 'bold';             % Bold font

hold on

% Plotting =================================================================
% Plot left dendrogram
% tree1 = linkage(Data, 'average');
tree1 = Z;
[treeHdl1, ~, order1] = dendrogram(axTree1, tree1, 0, 'Orientation', 'left');

% Set dendrogram color
set(treeHdl1, 'Color', [0, 0, 0]);
set(treeHdl1, 'LineWidth', 1.5);

% Refine left dendrogram axes
set(axTree1, 'XColor', 'none', ...
             'YColor', 'none', ...
             'YDir', 'reverse', ...
             'XTick', [], ...
             'YTick', [], ...
             'YLim', [1, rows] + [-0.5, 0.5]);

% Plot top dendrogram
% tree2 = linkage(Data.', 'average');
tree2 = Z;
[treeHdl2, ~, order2] = dendrogram(axTree2, tree2, 0, 'Orientation', 'top');
set(treeHdl2, 'Color', [0, 0, 0]);
set(treeHdl2, 'LineWidth', 1.5);

% Refine top dendrogram axes
YLimRange = get(axTree2, 'YLim'); % Get current range
set(axTree2, 'XColor', 'none', ...
             'YColor', 'none', ...
             'XTick', [], ...
             'YTick', [], ...
             'XLim', [1, cols] + [-0.5, 0.5]);

% Plot central heatmap
Data = Data(order1, :);
Data = Data(:, order2);
imagesc(axMain, Data);
axMain.XTickLabel = colName(order2);
axMain.YTickLabel = rowName(order1);

% Draw white grid lines
LineX = repmat([[1, cols] + [-.5, .5], nan], [rows+1, 1]).';
LineY = repmat((.5:1:(rows+.5)).', [1, 3]).';
plot(axMain, LineX(:), LineY(:), 'Color', 'w', 'LineWidth', 1);

LineY = repmat([[1, rows] + [-.5, .5], nan], [cols+1, 1]).';
LineX = repmat((.5:1:(cols+.5)).', [1, 3]).';
plot(axMain, LineX(:), LineY(:), 'Color', 'w', 'LineWidth', 1);

% Adjust colormap
baseCM = {[189, 53, 70; 255, 255, 255; 97, 97, 97]./255, ...
          [13, 135, 169; 255, 255, 255; 239, 139, 14]./255, ...
          [28, 127, 119; 255, 255, 255; 204, 157, 80]./255, ...
          [130, 130, 255; 255, 255, 255; 255, 133, 133]./255, ...
          [209, 58, 78; 253, 203, 121; 254, 254, 189; 198, 230, 156; 63, 150, 181]./255, ...
          [243, 166, 72; 255, 255, 255; 133, 121, 176]./255};

Cmap = baseCM{2};
Ci = 1:size(Cmap, 1);
Cq = linspace(1, size(Cmap, 1), 200); % Interpolate to 200 rows
Cmap = [interp1(Ci, Cmap(:,1), Cq, 'linear')' ...
        interp1(Ci, Cmap(:,2), Cq, 'linear')' ...
        interp1(Ci, Cmap(:,3), Cq, 'linear')'];

colormap(Cmap);
clim([min(min(Data)), max(max(Data))]);