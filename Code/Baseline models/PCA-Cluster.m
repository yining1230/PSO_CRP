clear
clc

% load datasig_q;
% load data_Pd_all_23-44;
load data_170_rd.mat data;

[data1,textdata1,raw1]=xlsread('\170_sigman2024Enantioselective Sulfonimidamide Acylation via a Cinchona.xlsx','Sheet3','G1:M171');
Designation = raw1(2:end, 7);
idx_train = strcmp(Designation, 'train'); 
idx_test = strcmp(Designation, 'test');   
idx_external = strcmp(Designation, 'external validation');

true_train_indices = find(idx_train);
true_test_indices = find(idx_test);
true_external_indices = find(idx_external);

trainData = data(idx_train, :);
validationData = data(idx_test, :);
% validationData = data(idx_external, :);

data = trainData;
[data1,textdata1,raw1]=xlsread('sigman_170.xlsx','conversion','A522:C692');

trainData_idx = data1(idx_train,2);
data1 = trainData_idx;

for i=1:size(data,2)
    data_norm(:,i)=(data(:,i)-mean(data(:,i)))/std(data(:,i));
end

data = data_norm;

k = 5; 
[coefficients, score, ~, ~, explained] = pca(data);

top_coefficients = coefficients(:, 1:k);

reduced_data = data * top_coefficients;
data_pca = reduced_data;


% scatter(reduced_data(:, 1), reduced_data(:, 2)); 
% xlabel('Principal Component 1');
% ylabel('Principal Component 2');
% title('PCA Dimensionality Reduction');

%% Clusting
new_score = data_pca;
X = new_score;
k=2;
% i_julei=1;
% for i_julei=1:5
% for i_julei=5
for i_julei=1:5
    if i_julei==1
        %% Hierarchical clustering
        va_dist={'euclidean','seuclidean','cityblock','mahalanobis','minkowski','cosine'};%
        va_dist{1}
        va_link={'single','complete','average','weighted','centroid','median','ward'};%
        va_link{1}
        idx_zong1=[];
        for i_distance=1:6
            i_distance
            for j_linkmethod=1:7
                j_linkmethod
                idx_cengci(:,j_linkmethod)=clusterdata(X,'maxclust',k,'distance',va_dist{i_distance},'linkage',va_link{j_linkmethod});
                % idx_cengci(:,1)=clusterdata(X,'maxclust',k,'distance',va_dist{1},'linkage',va_link{1});

                %  save i_hidx;
                %                 figure;
                %                 %  scatter3(X(:,1),X(:,2),X(:,3),100,idx_cengci,'filled')
                %                 plot(X(idx_cengci==1,1),X(idx_cengci==1,2),'r.','MarkerSize',12)
                %                 hold on
                %                 plot(X(idx_cengci==2,1),X(idx_cengci==2,2),'b.','MarkerSize',12)
                %                 hold on
                %                 plot(X(idx_cengci==3,1),X(idx_cengci==3,2),'m.','MarkerSize',12)
                %                 hold on
                %                 plot(X(idx_cengci==4,1),X(idx_cengci==4,2),'g.','MarkerSize',12)
                %                 hold on
                %
                % %                 plot(C_kmeans(:,1),C_kmeans(:,2),'kx',...
                % %                     'MarkerSize',15,'LineWidth',3)
                %                 legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
                %                     'Location','NW')
                %                 title 'Cluster Assignments and Centroids in 4 Groups'
                %                 xlabel('PC1','fontsize',12);  
                %                 ylabel('PC2','fontsize',12);   
                %                 zlabel('PC3','fontsize',12);   
            end
            idx_zong1=cat(2,idx_zong1,idx_cengci);
        end
        % hidx=clusterdata(X,'maxclust',k,'distance',va_dist{1},'linkage',va_link{1});
    elseif i_julei==2
        %% k-means
        va_distk={'sqeuclidean','cityblock'}%'hamming''cos''correlation'
        va_distk{1}
        % [idx,C] = kmeans(X,4);
        % [idx,C] = kmeans(X,4,"Distance","cityblock");
        % [idx,C] = kmeans(X,4,"Distance","cosine");
        % [idx,C] = kmeans(X,4,"Distance","hamming");
        % [idx,C] = kmeans(X,4,"Distance","correlation");
        for i_distancek=1:2
            i_distancek
            [idx_kmeans(:,i_distancek),C_kmeans]=kmeans(X,k,"Distance",va_distk{i_distancek});
            %         save i_hidx;
            %             figure;
            % %             scatter3(X(:,1),X(:,2),X(:,3),100,idx_kmeans,'filled')
            %             plot(X(idx_kmeans==1,1),X(idx_kmeans==1,2),'r.','MarkerSize',12)
            %             hold on
            %             plot(X(idx_kmeans==2,1),X(idx_kmeans==2,2),'b.','MarkerSize',12)
            %             hold on
            %             plot(X(idx_kmeans==3,1),X(idx_kmeans==3,2),'m.','MarkerSize',12)
            %             hold on
            %             plot(X(idx_kmeans==4,1),X(idx_kmeans==4,2),'g.','MarkerSize',12)
            %             hold on
            %
            %             plot(C_kmeans(:,1),C_kmeans(:,2),'kx',...
            %                 'MarkerSize',15,'LineWidth',3)
            %             legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
            %                 'Location','NW')
            %             title 'Cluster Assignments and Centroids in 4 Groups'
            %             xlabel('PC1','fontsize',12);
            %             ylabel('PC2','fontsize',12); 
            %             zlabel('PC3','fontsize',12); 
            idx_zong2=idx_kmeans;
        end

        % [~,idx_test] = pdist2(C,Xtest,'euclidean','Smallest',1);
        % gscatter(Xtest(:,1),Xtest(:,2),idx_test,'bgm','ooo')
        % legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid', ...
        %     'Data classified to Cluster 1','Data classified to Cluster 2', ...
        %     'Data classified to Cluster 3');

    elseif i_julei==3
        %% k-Medoids
        % idx = kmedoids(X,k);
        va_distkd={'sqEuclidean','euclidean','seuclidean','cityblock','minkowski','mahalanobis','chebychev','cosine','correlation','spearman','hamming','jaccard'}%12种距离公式
        va_distkd{1}

        va_Algorithm={'pam','small','clara','large'}
        va_Algorithm{1}
        %         [idx_kmedoid,C_kmedoid,sumd_kmedoid,d_kmedoid,midx_kmedoid,info_kmedoid] = kmedoids(X,4,'Distance',va_distkd{1},'Algorithm',va_Algorithm{i});%第二种方法
        idx_zong3=[];
        for i_distancekd=1:12
            i_distancekd
            for m_Algorithm=1:4
                m_Algorithm
                [idx_kmedoid(:,m_Algorithm),C_kmedoid,sumd_kmedoid,d_kmedoid,midx_kmedoid,info_kmedoid] = kmedoids(X,k,'Distance',va_distkd{i_distancekd},'Algorithm',va_Algorithm{m_Algorithm});%第二种方法
                %         save i_hidx;
                %                 figure;
                % %                 scatter3(X(:,1),X(:,2),X(:,3),100,idx_kmedoid,'filled')
                %                 plot(X(idx_kmedoid==1,1),X(idx_kmedoid==1,2),'r.','MarkerSize',12)
                %                 hold on
                %                 plot(X(idx_kmedoid==2,1),X(idx_kmedoid==2,2),'b.','MarkerSize',12)
                %                 hold on
                %                 plot(X(idx_kmedoid==3,1),X(idx_kmedoid==3,2),'m.','MarkerSize',12)
                %                 hold on
                %                 plot(X(idx_kmedoid==4,1),X(idx_kmedoid==4,2),'g.','MarkerSize',12)
                %                 plot(C_kmedoid(:,1),C_kmedoid(:,2),'kx',...
                %                     'MarkerSize',15,'LineWidth',3)
                %                 legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Medoids',...
                %                     'Location','NW');
                %                 title('Cluster Assignments and Medoids');
                %                 hold off
            end
            idx_zong3=cat(2,idx_zong3,idx_kmedoid);
        end
        % opts = statset('Display','iter');
        % [idx,C,sumd,d,midx,info] = kmedoids(X,4,'Distance','cityblock','Options',opts);
        % [idx,C,sumd,d,midx,info] = kmedoids(X,4,'Distance','hamming','Options',opts);
        % [idx,C,sumd,d,midx,info] = kmedoids(X,4,'Distance','hamming','Algorithm','pam');
        % [idx,C,sumd,d,midx,info] = kmedoids(X,4,'Distance','hamming','Algorithm','small');

    elseif i_julei==4 
        %% GMM
        % GMModel = fitgmdist(X,k);
        % idx_GMM = cluster(GMModel,X)
        % idx = posterior(GMModel,X);
        % idx = mahal(GMModel,X);
        % GMModel = fitgmdist(X,2, 'RegularizationValue' ,0.1);
        for i_GMM=2
            if i_GMM==1
                GMModel = fitgmdist(X,k);
                idx_GMM = cluster(GMModel,X);
            else
                GMModel = fitgmdist(X,k,'RegularizationValue' ,0.1);
                idx_GMMn = cluster(GMModel,X);
            end
            %             figure;
            %             plot(X(idx_GMM==1,1),X(idx_GMM==1,2),'r.','MarkerSize',12)
            %             hold on
            %             plot(X(idx_GMM==2,1),X(idx_GMM==2,2),'b.','MarkerSize',12)
            %             hold on
            %             plot(X(idx_GMM==3,1),X(idx_GMM==3,2),'m.','MarkerSize',12)
            %             hold on
            %             plot(X(idx_GMM==4,1),X(idx_GMM==4,2),'g.','MarkerSize',12)
            %             hold on
            %             %     plot(C_kmeans(:,1),C_kmeans(:,2),'kx',...
            %             %         'MarkerSize',15,'LineWidth',3)
            %             legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids',...
            %                 'Location','NW')
            %             title 'Cluster Assignments and Centroids in 4 Groups'
            %             xlabel('PC1','fontsize',12);  
            %             ylabel('PC2','fontsize',12);  
            %             zlabel('PC3','fontsize',12);  
        end
        %         idx_zong4=cat(2,idx_GMM,idx_GMMn);
        idx_zong4=idx_GMMn;
    else
        %% spectral clustering
        % idx = spectralcluster(X,k);
        % idx = spectralcluster(X,k,"Distance","chebychev");
        % idx = spectralcluster(X,k,"Distance","chebychev","ClusterMethod","kmeans");
        %         va_dists={"euclidean","seuclidean","mahalanobis","cityblock","minkowski","chebychev","cosine","correlation","hamming","jaccard","spearman"}%去掉"precomputed",
        va_dists={"euclidean","seuclidean","mahalanobis","cityblock","minkowski","chebychev","cosine","correlation","spearman"}%去掉"precomputed","hamming"，"jaccard",

        va_dists{1}

        va_ClusterMethod={"kmeans","kmedoids"}
        va_ClusterMethod{1}
        % idx_sc = spectralcluster(X,k,"Distance",va_dists{7},"ClusterMethod",va_ClusterMethod{1});
        idx_zong5=[];
        for i_distances=1:9
            i_distances
            for n_ClusterMethod=1:2
                n_ClusterMethod
                idx_sc(:,n_ClusterMethod) = spectralcluster(X,k,"Distance",va_dists{i_distances},"ClusterMethod",va_ClusterMethod{n_ClusterMethod});
                %         save i_hidx;
                %                 figure;
                %                 plot(X(idx_sc==1,1),X(idx_sc==1,2),'r.','MarkerSize',12)
                %                 hold on
                %                 plot(X(idx_sc==2,1),X(idx_sc==2,2),'b.','MarkerSize',12)
                %                 hold on
                %                 plot(X(idx_sc==3,1),X(idx_sc==3,2),'m.','MarkerSize',12)
                %                 hold on
                %                 plot(X(idx_sc==4,1),X(idx_sc==4,2),'g.','MarkerSize',12)
                %                 %         plot(C(:,1),C(:,2),'kx',...
                %                 %             'MarkerSize',15,'LineWidth',3)
                %                 legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Medoids',...
                %                     'Location','NW');
                %                 title('Cluster Assignments and Medoids');
                %                 hold off
            end
            idx_zong5=cat(2,idx_zong5,idx_sc);
        end
    end
end
% idx_zong=idx_zong1;
% idx_zong=cat(2,idx_zong1,idx_zong2);
% idx_zong=idx_zong5;

idx_zong=cat(2,idx_zong1,idx_zong2,idx_zong3,idx_zong4,idx_zong5);
for j=1:size(idx_zong,2)

    idx_julei=idx_zong(:,j)
    b_julei=idx_julei;
    b_julei(1);
      m1=size(b_julei,1);
    k
    n1=k;
    A_julei=zeros(m1,n1); 

    for i=1:m1
        A_julei(i,b_julei(i))=1
    end

    dlY = dlarray(A_julei,'CB');

    size(dlY) 
    dims(dlY)

    m=size(dlY,1);
    n=size(dlY,2);
    A_fenlei=zeros(m,n); 

    % [data1,textdata1,raw1]=xlsread('\aaaa.xlsx','sheet5','H1:L48');
    % data1 = evalin('base', 'data1');

    b_fenlei=data1(:,1);
    b_fenlei(1)

    for i=1:size(data1,1)
        A_fenlei(i,b_fenlei(i))=1
    end
    targets=A_fenlei;
    % [data1,textdata1,raw1]=xlsread('\aaaa.xlsx','sheet5','H1:K48');
    % targets=data1;
    targets = single(targets);
    size(targets)

    loss(j) = crossentropy(dlY,targets,'TargetCategories','independent')
    %     loss(1)
    Loss=double(loss);

end
% fitness=Loss;
% fitness=min(Loss);
[fitness, fitnessindex]=min(Loss);
fitnessclu=idx_zong(:,fitnessindex);

% X_datanorm=data_norm;
E = extractdata(fitnessindex)
if E<=42
    m1=fix(E/size(va_link,2));
    m2=E-m1*size(va_link,2);
    m1=m1+1;
    va_dist{m1};
    if m2==0
        Me1=va_link{size(va_link,2)}
    else
        Me1=va_link{m2}
    end
    method={va_dist{m1} Me1};
    i_julei=1;
elseif E<=44
    m=44-E;
    method=va_distk{m};
    i_julei=2;
elseif E<=92
    m=E-44;
    n1=fix(m/size(va_Algorithm,2));
    n2=m-n1*size(va_Algorithm,2);
    n1=n1+1;
    va_distkd{n1};
    if n2==0
        Me3=va_Algorithm{size(va_Algorithm,2)}
    else
        Me3=va_Algorithm{n2};
    end
        method={va_distkd{n1} Me3};
        i_julei=3;
elseif E==93
    method=GMModel_zhengze;
    i_julei=4;
else 
    m=E-93;
    n1=fix(m/size(va_ClusterMethod,2));
    n2=m-n1*size(va_ClusterMethod,2);
    n1=n1+1;
    va_dists{n1};
    if n2==0
        Me5=va_ClusterMethod{size(va_ClusterMethod,2)}
    else
        Me5=va_ClusterMethod{n2};
    end
    method={va_dists{n1} Me5};
    i_julei=5;
end
method

%% predictions on new data
% [datap,textdatap,rawp]=xlsread('noise_0.03.xlsx','Sheet1','A1:H11');
datap = validationData;
datap_norm = [];

for i=1:size(datap,2)
    datap_norm(:,i)=(datap(:,i)-mean(datap(:,i)))/std(datap(:,i));
end
datap = datap_norm;
inputp_dim = size(datap, 2); 
encodingp_dim = 5; 

% Create an autoencoder model.
autoencoderp = trainAutoencoder(datap', encodingp_dim); 

encoded_datap = encode(autoencoderp, datap');
encoded_datap = encoded_datap';

disp(encoded_data);
datap_ = encoded_datap;

% shuffledIndices = shuffledIndices';

%% Obtain centroids from hierarchical clustering.
uniqueLabels = unique(fitnessclu);

centroids = zeros(length(uniqueLabels), size(data_pca, 2)); 
for i = 1:length(uniqueLabels)
clusterData = data_pca(fitnessclu == uniqueLabels(i), :); 
centroids(i, :) = mean(clusterData); 
end
C_centroids=centroids;
distances = pdist2(datap_pca, C_centroids,'seuclidean'); 

newC = knnsearch(C_centroids, datap_pca); 
disp(newC);
idx_pre=newC;


%% kmedoid
unique_labels = unique(fitnessclu);

% Initialize variables to store the centroid of each cluster.
centroids = zeros(length(unique_labels), size(data_pca, 2)); 
% For each cluster label, find the centroid.
for i = 1:length(unique_labels)
    cluster_indices = find(fitnessclu == unique_labels(i));
    
    if exist('centroid_indices', 'var')
        centroid_index = centroid_indices(i);
        centroids(i, :) = data_pca(centroid_index, :);
    else
        
        cluster_data = data_pca(cluster_indices, :);
        distances = pdist2(cluster_data, cluster_data, 'squaredeuclidean'); 
        [~, min_distance_index] = min(mean(distances, 2)); 
        centroids(i, :) = cluster_data(min_distance_index, :); 
    end
end

disp(centroids);

predicted_labels = zeros(size(datap_pca, 1), 1);

% For each new sample, compute its distance to each centroid and assign it to the nearest class.
for i = 1:size(datap_pca, 1)
    distances = pdist2(datap_pca(i, :), centroids, 'squaredeuclidean');
    [~, min_distance_index] = min(distances);
    predicted_labels(i) = min_distance_index; 
end
idx_pre =  predicted_labels;

%% kmedoid
unique_labels = unique(fitnessclu);
centroids = zeros(length(fitnessclu), size(data_pca, 2)); 

centroids = zeros(length(unique_labels), size(data_pca, 2));

% For each cluster label, find the centroid.
for i = 1:length(unique_labels)
    cluster_indices = find(fitnessclu == unique_labels(i));
    
    % Compute the mean and covariance matrix of the clustered data.
    cluster_data = data_pca(cluster_indices, :);
    cluster_mean = mean(cluster_data);
    cluster_covariance = cov(cluster_data);
    
    
    mahalanobis_distances = sqrt(sum(((cluster_data - cluster_mean) / cluster_covariance) .* (cluster_data - cluster_mean), 2));
    
    [~, min_distance_index] = min(mahalanobis_distances);
    centroids(i, :) = cluster_data(min_distance_index, :);
end

disp(centroids);
predicted_labels = zeros(size(datap_pca, 1), 1);

for i = 1:size(datap_pca, 1)
    mahalanobis_distances = zeros(length(unique_labels), 1); 
    for j = 1:2 
        x = datap_pca(i, :); 
        mu = centroids(j, :); 
        covariance_matrix = cov(cluster_data); 
        inverse_covariance_matrix = inv(covariance_matrix); 
        mahalanobis_distance = sqrt((x - mu) * inverse_covariance_matrix * (x - mu)'); 
        mahalanobis_distances(j) = mahalanobis_distance; 
    end
    [~, min_distance_index] = min(mahalanobis_distances); 
    predicted_labels(i) = min_distance_index; 
end
idx_pre =  predicted_labels;




%% GMM
unique_labels = unique(fitnessclugbesthuizong);

num_clusters = numel(unique_labels);
gmm_models = cell(num_clusters, 1);
k=3;
for i = 1:num_clusters
    cluster_data = X(fitnessclugbesthuizong == unique_labels(i), :);
    gmm_model = fitgmdist(cluster_data, k,'RegularizationValue' ,0.1);
    gmm_models{i} = gmm_model;
end

% % newdata = [newdata1; newdata2; ...];   
% % posterior = posterior(GMModel, newdata);   
% % [~, predLabels] = max(posterior, [], 2);   
% clusterIdx = cluster(gmm_models, datap_kpca);
% disp(clusterIdx);
unique_labels = unique(fitnessclu);
newData = datap_pca;

predictedLabels = zeros(size(newData, 1), 1); 

for i = 1:size(newData, 1)
    likelihoods = zeros(length(unique_labels), 1);

    for j = 1:length(unique_labels)
        likelihoods(j) = pdf(gmm_models{j}, newData(i, :));
    end
    
    [~, predictedLabels(i)] = max(likelihoods);
end
idx_pre=predictedLabels;


%% spectral clustering
similarities_new = pdist2(datap_pca, data_pca,'mahalanobis');
 [~, predicted_labels] = min(similarities_new * sparse(1:numel(fitnessclu), fitnessclu, 1), [], 2); 
idx_pre=predicted_labels;
% similarities_new = pdist2(datap_pca, data_pca,'mahalanobis');
%  [~, predicted_labels] = min(similarities_new * sparse(1:numel(fitnessclu), fitnessclu, 1), [], 2); 
% idx_pre=predicted_labels;
unique_labels = unique(fitnessclu);
centroids = zeros(length(unique_labels), size(data_pca, 2));
for i = 1:length(unique_labels)
    cluster_data = data_pca(data_pca == unique_labels(i), :);
    centroids(i, :) = mean(cluster_data);
end

predicted_labels = zeros(size(datap_pca, 1), 1);
for i = 1:size(datap_pca, 1)
    distances = pdist2(datap_pca(i, :), centroids, 'correlation');
    [~, min_distance_index] = min(distances);
    predicted_labels(i) = min_distance_index;
end
idx_pre1 = predicted_labels;

