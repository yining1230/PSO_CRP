function [fitness,fitnessindex,fitnessclu,X_datanorm]=fun_fitnesscelclu(data,xx) 
for i=1:size(data,2)
    data_norm(:,i)=(data(:,i)-mean(data(:,i)))/std(data(:,i));
end
% xx=x(i,:);
% xx=x(1);

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

K = kernel_matrix(data_norm, va_kerneltype{i_kerneltype}, struct);
% struct.degree=2;
% struct.constant=1;
% struct.sigma=1;
% K = kernel_matrix(data_norm, 'rq', struct);
% 
% K = kernel_matrix(data_norm, 'polynomial',struct);

n = size(K,1);
one_mat = ones(n,n)/n;
K_centered = K - one_mat*K - K*one_mat + one_mat*K*one_mat;

[V, D] = eig(K_centered);
[D_sorted, idx] = sort(diag(D),'descend'); 

V_sorted = V(:,idx);


% for i=1:size(data,2)
%     D_sorted(i,2)=D_sorted(i,1)/sum(D_sorted(:,1));
%     D_sorted(i,3)=sum(D_sorted(1:i,1))/sum(D_sorted(:,1));
% end


k_t=5
alpha_k = V_sorted(:,1:k_t);

data_kpca = K_centered * alpha_k;


new_score=data_kpca;
X=new_score;
k=2;
% i_julei=1;
% for i_julei=1:5
% for i_julei=5
for i_julei=1:5
    if i_julei==1
        %% Hierarchical clustering
        va_dist={'euclidean','seuclidean','mahalanobis','cityblock','minkowski','cosine'};%
        va_dist{1}
        va_link={'single','complete','average','weighted','centroid','median','ward'};%
        va_link{1}
        idx_zong1=[];
        for i_distance=1:6
            i_distance
            for j_linkmethod=1:7
                j_linkmethod
                idx_cengci(:,j_linkmethod)=clusterdata(X,'maxclust',k,'distance',va_dist{i_distance},'linkage',va_link{j_linkmethod});
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
        va_distkd={'sqEuclidean','euclidean','seuclidean','cityblock','minkowski','chebychev','mahalanobis','cosine','correlation','spearman','hamming','jaccard'}
        va_distkd{1}

        va_Algorithm={'pam','small','large'}%3'clara',
        va_Algorithm{1}
        %         [idx_kmedoid,C_kmedoid,sumd_kmedoid,d_kmedoid,midx_kmedoid,info_kmedoid] = kmedoids(X,4,'Distance',va_distkd{1},'Algorithm',va_Algorithm{i});
        idx_zong3=[];
        for i_distancekd=1:12
            i_distancekd
            for m_Algorithm=1:3
                m_Algorithm
                [idx_kmedoid(:,m_Algorithm),C_kmedoid,sumd_kmedoid,d_kmedoid,midx_kmedoid,info_kmedoid] = kmedoids(X,k,'Distance',va_distkd{i_distancekd},'Algorithm',va_Algorithm{m_Algorithm});
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
            %             xlabel('PC1','fontsize',12);   % x轴的名称
            %             ylabel('PC2','fontsize',12);   % y轴的名称
            %             zlabel('PC3','fontsize',12);   % y轴的名称
        end
        %         idx_zong4=cat(2,idx_GMM,idx_GMMn);
        idx_zong4=idx_GMMn;
    else
        %% spectral clustering
        % idx = spectralcluster(X,k);
        % idx = spectralcluster(X,k,"Distance","chebychev");
        % idx = spectralcluster(X,k,"Distance","chebychev","ClusterMethod","kmeans");
%         va_dists={"euclidean","seuclidean","mahalanobis","cityblock","minkowski","chebychev","cosine","correlation","hamming","jaccard","spearman"}%"precomputed",
        va_dists={"euclidean","seuclidean","mahalanobis","cityblock","minkowski","chebychev","cosine","correlation","spearman"}%"precomputed","hamming"，"jaccard",

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

    % [data1,textdata1,raw1]=xlsread('C:\Users\DELL\Desktop\aaaa.xlsx','sheet5','H1:L48');
    data1 = evalin('base', 'data1');

    b_fenlei=data1(:,1);
    b_fenlei(1)

    for i=1:size(data1,1)
        A_fenlei(i,b_fenlei(i))=1
    end
    targets=A_fenlei;
    % [data1,textdata1,raw1]=xlsread('C:\Users\29868\Desktop\aaaa.xlsx','sheet5','H1:K48');
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
X_datanorm=data_norm;
% fitness=L'
end
