X=data_kpca

%% GMM
idx_GMMn=fitnessclugbesthuizong;
figure;
plot(X(idx_GMMn==1,1),X(idx_GMMn==1,2),'r.','MarkerSize',12)
hold on 
plot(X(idx_GMMn==2,1),X(idx_GMMn==2,2),'b.','MarkerSize',12)
hold on
plot(X(idx_GMMn==3,1),X(idx_GMMn==3,2),'m.','MarkerSize',12)
hold on
% plot(X(idx_GMM==4,1),X(idx_GMM==4,2),'g.','MarkerSize',12)
% hold on
%     plot(C_kmeans(:,1),C_kmeans(:,2),'kx',...
%         'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
    'Location','NW')
title 'Cluster Assignments and Centroids in 3 Groups'
xlabel('PC1','fontsize',12);  
ylabel('PC2','fontsize',12);  
zlabel('PC3','fontsize',12);  
% end
%% gmm

unique_labels = unique(fitnessclugbesthuizong);
num_clusters = numel(unique_labels);
gmm_models = cell(num_clusters, 1);

k=3;
for i = 1:num_clusters
    cluster_data = X(fitnessclugbesthuizong == unique_labels(i), :);
    gmm_model = fitgmdist(cluster_data, k,'RegularizationValue' ,0.1);
    gmm_models{i} = gmm_model;
end


%% k-Medoids
va_distkd={'sqEuclidean','euclidean','seuclidean','cityblock','minkowski','chebychev','mahalanobis','cosine','correlation','spearman','hamming','jaccard'}
va_distkd{4}

va_Algorithm={'pam','small','clara','large'}
va_Algorithm{2}
[idx_kmedoid,C_kmedoid,sumd_kmedoid,d_kmedoid,midx_kmedoid,info_kmedoid] = kmedoids(X,3,'Distance',va_distkd{6},'Algorithm',va_Algorithm{2});
%         save i_hidx;

figure;
%                 scatter3(X(:,1),X(:,2),X(:,3),100,idx_kmedoid,'filled')
plot(X_kpca(fitnessclugbesthuizong==1,1),X_kpca(fitnessclugbesthuizong==1,2),'r.','MarkerSize',12)
hold on
plot(X_kpca(fitnessclugbesthuizong==2,1),X_kpca(fitnessclugbesthuizong==2,2),'b.','MarkerSize',12)
hold on
plot(X_kpca(fitnessclugbesthuizong==3,1),X_kpca(fitnessclugbesthuizong==3,2),'m.','MarkerSize',12)
hold on
plot(C_kmedoid(:,1),C_kmedoid(:,2),'k*',...
     'MarkerSize',7,'LineWidth',1.5)
% plot(X_kpca(fitnessclugbesthuizong==4,1),X_kpca(fitnessclugbesthuizong==4,2),'g.','MarkerSize',12)
%     plot(C_kmedoid(:,1),C_kmedoid(:,2),'kx',...
%     %     'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Cluster 3','Medoids',...
    'Location','NW');
title('Cluster Assignments and Medoids');
hold off
xlabel('kPC1','fontsize',12);   
ylabel('kPC2','fontsize',12);   
zlabel('kPC3','fontsize',12);   



%% predictions on new data.

% [datap,textdatap,rawp]=xlsread('C:\Users\DELL\Desktop\10noise_0.03.xlsx','Sheet1','A1:H11');
datap = validationData;

xx(1)=gbest;
va_kerneltype={'polynomial','gaussian','laplaceRBF','anova','rq','mk','imk'};
i_kerneltype = evalin('base', 'i_kerneltype');
if i_kerneltype==1
    struct.degree=xx(1)
    struct.constant=xx(2)
elseif  i_kerneltype==2
    struct.sigma=xx(1)
elseif  i_kerneltype==3
    struct.sigma=xx(1)
elseif  i_kerneltype==4
    struct.sigma=xx(1)
    struct.degree=xx(2)
elseif  i_kerneltype==5
    struct.constant=xx(1)
elseif  i_kerneltype==6
    struct.constant=xx(1)
else
    struct.constant=xx
end    

K_p = kernel_matrix(datap, va_kerneltype{i_kerneltype}, struct);
% K(1,2)=exp(-norm(datap_norm(1,:)-datap_norm(2,:))^2/(2*gbest^2));
% K(1,4)=exp(-norm(data_norm(1,:)-data_norm(4,:))^2/(2*gbest^2));
% sum(abs((datap_norm(1,:)-datap_norm(2,:)).^2))
% m=abs((datap_norm(1,:)-datap_norm(2,:)).^2)
% sum(m)
% struct.degree=2;
% struct.constant=1;
% struct.sigma=1;
% K = kernel_matrix(data_norm, 'rq', struct);
% 
% K = kernel_matrix(data_norm, 'polynomial',struct);


n = size(K_p,1);
one_mat = ones(n,n)/n;
Kp_centered = K_p - one_mat*K_p - K_p*one_mat + one_mat*K_p*one_mat;

[Vp, Dp] = eig(Kp_centered);
[Dp_sorted, idx_p] = sort(diag(Dp),'descend'); 
Vp_sorted = Vp(:,idx_p);

% for i=1:size(datap,2)
%     Dp_sorted(i,2)=Dp_sorted(i,1)/sum(Dp_sorted(:,1));
%     Dp_sorted(i,3)=sum(Dp_sorted(1:i,1))/sum(Dp_sorted(:,1));
% end

% k_t=5
alpha_k = Vp_sorted(:,1:k_t);
datap_kpca = Kp_centered * alpha_k;
% shuffledIndices=shuffledIndices';

%% spectral clustering
uniqueLabels = unique(fitnessclugbesthuizong);
centroids = zeros(length(uniqueLabels), size(data_kpca, 2)); 
for i = 1:length(uniqueLabels)
clusterData = data_kpca(fitnessclugbesthuizong == uniqueLabels(i), :); 
centroids(i, :) = mean(clusterData); 
end
C_centroids=centroids;
%% spectral clustering
distances = pdist2(datap_kpca, C_centroids,'euclidean'); 
newC = knnsearch(C_centroids, datap_kpca); 

disp(newC);
idx_pre=newC;


%% kmedoid
% [~,idx_pre] = pdist2(C_kmedoid,datap_kpca,'seuclidean','small',1);
[~,idx_pre] = pdist2(C_kmedoid,datap_kpca,'cosine','large',1);

idx_pre = idx_pre';

%% GMM
% % newdata = [newdata1; newdata2; ...];  
% % posterior = posterior(GMModel, newdata);   
% % [~, predLabels] = max(posterior, [], 2);   
% clusterIdx = cluster(gmm_models, datap_kpca);
% disp(clusterIdx);

newData = datap_kpca; 

predictedLabels = zeros(size(newData, 1), 1);
unique_labels = unique(fitnessclugbesthuizong);

num_clusters = numel(unique_labels);
gmm_models = cell(num_clusters, 1);

k=2;
for i = 1:num_clusters
    cluster_data = X(fitnessclugbesthuizong == unique_labels(i), :);
    gmm_model = fitgmdist(cluster_data, k,'RegularizationValue' ,0.1);
    gmm_models{i} = gmm_model;
end

for i = 1:size(newData, 1)
    likelihoods = zeros(length(unique_labels), 1);
    
    for j = 1:length(unique_labels)
        likelihoods(j) = pdf(gmm_models{j}, newData(i, :));
    end
    
    [~, predictedLabels(i)] = max(likelihoods);
end
idx_pre=predictedLabels;

%%spectral clustering 
similarities_new = pdist2( datap_kpca, data_kpca,'cosine');

 [~, predicted_labels] = min(similarities_new * sparse(1:numel(fitnessclugbesthuizong), fitnessclugbesthuizong, 1), [], 2); 
idx_pre=predicted_labels;


hold on

gscatter(datap_kpca(:,1),datap_kpca(:,2),idx_pre,'rbm','o')
legend('Cluster 1','Cluster 2','Cluster 3', ...
    'Data classified to Cluster 1','Data classified to Cluster 2', ...
    'Data classified to Cluster 3')
legend('Cluster 1','Cluster 2', ...
    'Data classified to Cluster 1','Data classified to Cluster 2')