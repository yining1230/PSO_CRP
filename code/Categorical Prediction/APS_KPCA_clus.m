%% This code implements function extremum optimization based on CDW-PSO—iterative optimization of kernel function parameters in Kernel Principal Component Analysis (KPCA).


%         for i_kerneltype=1:8
%             i_kerneltype
%         end
clc
clear all;

% load datasz %sigman
% [data,textdata,raw]=xlsread('C:\Users\DELL\Desktop\hte\s18_para.xlsx','s18_para','A1:DQ44');
% save datas18;
% [data,textdata,raw]=xlsread('C:\Users\DELL\Desktop\hte\s3_para.xlsx','s3_paraq','A1:DA43');
% save datas3q;

load data_170;
rand('seed',10); 

dataSize = size(data,1);

trainSize = round(0.7 * dataSize);
validationSize = dataSize - trainSize; 

shuffledIndices = randperm(dataSize);

trainData = data(shuffledIndices(1:trainSize),:);
validationData = data(shuffledIndices(trainSize+1:end),:);
data = trainData;

[data1,textdata1,raw1]=xlsread('E:\CNN\hte_170_sigman\sigman_170.xlsx','conversion','A1:C171');
% [data1,textdata1,raw1]=xlsread('C:\Users\DELL\Desktop\aaaa.xlsx','sheet8','Y1:Z26');
%         [data1,textdata1,raw1]=xlsread('C:\Users\29868\Desktop\aaaa.xlsx','sheet6','A1:B38');
trainData_idx = data1(shuffledIndices(1:trainSize),2);
data1 = trainData_idx;
%% 
% va_kerneltype={'polynomial','gaussian','laplaceRBF','anova','rq','mk','imk'}
% va_kerneltype{3}
i_kerneltype=2;
if i_kerneltype==1
    D=2
elseif  i_kerneltype==2
    D=1
elseif  i_kerneltype==3
    D=1
elseif  i_kerneltype==4
    D=2
elseif  i_kerneltype==5
    D=1
elseif  i_kerneltype==6
    D=1
else
    D=1
end    

c1 = 2; 
c2 = 2;

maxgeneration=100; 
si=40;  


xmax=3; 
xmin=1; 
vmax=xmax/10; 
vmin=xmin/10; 
rmax=1; 
rmin=0; 
w=rand; 
% load datasig_q;
% i=1
% si=2;   
% 

%% Generate initial particles and velocities (initialization)
for i=1:si

    x(i,:)=xmax*rands(1,D);   
    v(i,:)=vmax*rands(1,D); 

%     x_xishu=x(i,:); 
%     save x_xishu;
%     fitness(i,:)=fun_fitnesscel(data,x(i,:));   
%     fitness(:,i)=fun_fitnesscel(data,x(i,:));  
    [fitness(i),fitnessindex(i),fitnessclu(:,i),X_datanorm]=fun_fitnesscelclu(data,x(i,:));   

end
%% Personal best and global best
[bestfitness, bestindex]=min(fitness);
pbest=x;  
gbest=x(bestindex,:);   
fitnesspbest=fitness;   
fitnessgbest=bestfitness;   
fitnessindexgbest= bestindex;
fitnessclugbest=fitnessclu(:,bestindex);

%% Iterative optimization

% maxgeneration=3;  
% i=1;

for i=1:maxgeneration
    i
    r=rmax-(rmax-rmin)*((i-1)/(maxgeneration-1));
    %  r=abs(rmax-(rmax-rmin)*exp(0.35)/(exp((i-1)/(maxgeneration-1))^0.35));
    %      r=rmax-(rmax-rmin)*((i-1)/(maxgeneration-1))^(1/3);%可以
    %   r=rmax-(rmax-rmin)*exp(1-maxgeneration/i);%不行
    
    
    w=r*sin(pi*w);
    u=mean(fitness);
    
    for j=1:si
        
        v(j,:) = w*v(j,:) + c1*rand*(pbest(j,:) - x(j,:)) + c2*rand*(gbest - x(j,:));
        v(j,find(v(j,:)>vmax))=vmax;
        v(j,find(v(j,:)<vmin))=vmin;
        
        %Population update
        u(i)=mean(fitness);
        w1=exp(fitness(j)/u(1))/(1+exp(-(fitness(j)/u(1))))^i;
        w2=1-w1;
        delta=w1;
        
        
        x(j,:)=w1*x(j,:)+w2*v(j,:)+rand*gbest*delta;
        x(j,find(x(j,:)>xmax))=xmax;
        x(j,find(x(j,:)<xmin))=xmin;
%         x_xishu=x(j,:);
%         save x_xishu;
        % Fitness value
        [fitness(j),fitnessindex(j),fitnessclu(:,j),X_datanorm]=fun_fitnesscelclu(data,x(j,:));
       
    end
    
    for j=1:si
        
        % Update the personal best (or individual best).
        if fitness(j) < fitnesspbest(j)
            pbest(j,:) = x(j,:);
            fitnesspbest(j) = fitness(j);
            fitnessindexpbest(j)=fitnessindex(j);
            fitnessclupbest=fitnessclu(:,j)
        end
        
        % Update the global best (or swarm best).
        if fitness(j) < fitnessgbest
            gbest = x(j,:);
            fitnessgbest = fitness(j);
            fitnessindexgbest=fitnessindex(j);
            fitnessclugbest=fitnessclu(:,j)
        end
    end
    yy(i)=fitnessgbest;
    yyy(i)=fitnessindexgbest;
    yyyy(:,i)=fitnessclugbest;
    yyyyy=X_datanorm;
end

%% Save the minimum cross-entropy value and the corresponding parameter values.
if i_kerneltype==1
    gbest_shuchu1=gbest;
    save gbest_shuchu1
elseif i_kerneltype==2
    gbest_shuchu2=gbest;
    save gbest_shuchu2
elseif i_kerneltype==3
    gbest_shuchu3=gbest;
    save gbest_shuchu3
elseif i_kerneltype==4
    gbest_shuchu4=gbest;
    save gbest_shuchu4
elseif i_kerneltype==5
    gbest_shuchu5=gbest;
    save gbest_shuchu5
elseif i_kerneltype==6
    gbest_shuchu6=gbest;
    save gbest_shuchu6
elseif i_kerneltype==7
    gbest_shuchu7=gbest;
    save gbest_shuchu7
else
    gbest_shuchu8=gbest;
    save gbest_shuchu8
end

% gbest_shuchu(i_hanshuleixing,:)=gbest;
% save gbest_shuchu;

fitnessgbesthuizong=fitnessgbest;% fitnessbest 
fitnessindexgbesthuizong=fitnessindexgbest;
fitnessclugbesthuizong=fitnessclugbest;
X_datanorm;


% stdbiaozhun=std(gbest);
% semilogy(yy,'-m','LineWidth',1)
% xlabel('Iterations','fontsize',12);
% ylabel('Mean of Best Function Values','fontsize',12);
% hold on
% meanzuizhong=mean(fitnessgbesthuizong)
% stdzuizhong=std(fitnessgbesthuizong);
for i_julei=1:5
    if i_julei==1
        %% Hierarchical clustering
        va_dist={'euclidean','seuclidean','mahalanobis','cityblock','minkowski','cosine'}
        size(va_dist)
        va_dist{1}
        va_link={'single','complete','average','weighted','centroid','median','ward'};
        va_link{1}
         % hidx=clusterdata(X,'maxclust',k,'distance',va_dist{1},'linkage',va_link{1});
    elseif i_julei==2
        %% k-means
        va_distk={'sqeuclidean','cityblock'}%'hamming''cos''correlation'
        va_distk{1}

    elseif i_julei==3
        %% k-Medoids
        % idx = kmedoids(X,k);
        va_distkd={'sqEuclidean','euclidean','seuclidean','cityblock','minkowski','chebychev','mahalanobis','cosine','correlation','spearman','hamming','jaccard'}%12种距离公式
        va_distkd{1}

        va_Algorithm={'pam','small','large'}%'clara',
        va_Algorithm{1}
       
    elseif i_julei==4 %%
        %% GMM
        % GMModel = fitgmdist(X,k);
        % idx_GMM = cluster(GMModel,X)
        % idx = posterior(GMModel,X);
        % idx = mahal(GMModel,X);
        % GMModel = fitgmdist(X,2, 'RegularizationValue' ,0.1);
        for i_GMM=2
            if i_GMM==1
                meth="GMModel_zhengze";
%                 GMModel = fitgmdist(X,k);
%                 idx_GMM = cluster(GMModel,X);
            else
                meth="GMModel_zhengze";
% GMModel = fitgmdist(X,k,'RegularizationValue' ,0.1);
%                 idx_GMMn = cluster(GMModel,X);
            end
        end
        else
        %% spectral clustering
        va_dists={"euclidean","mahalanobis","cityblock","minkowski","chebychev","cosine"}%去掉"precomputed","hamming"，"jaccard","correlation","seuclidean",,"spearman"
        va_dists{1}

        va_ClusterMethod={"kmeans","kmedoids"}
        va_ClusterMethod{1}
    end
end
% fitnessindexpbesthuizong=36;
fitnessindexgbesthuizong
E = extractdata(fitnessindexgbesthuizong)
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
elseif E<=80
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
elseif E==81
    method=GMModel_zhengze;
    i_julei=4;
else 
    m=E-81;
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
i_julei

% for i=1:size(data,2)
% data_norm(:,i)=(data(:,i)-mean(data(:,i)))/std(data(:,i));
% end

X_datanorm;
va_kerneltype={'polynomial','gaussian','laplaceRBF','anova','rq','mk','imk'};
i_kerneltype
canshu=extractdata(fitnessindexgbesthuizong);

va_kerneltype{i_kerneltype}
i_kerneltype=2
if i_kerneltype==1
    struct.degree=canshu(1)
    struct.constant=canshu(2)
elseif  i_kerneltype==2
    struct.sigma=canshu(1)
elseif  i_kerneltype==3
    struct.sigma=canshu(1)
elseif  i_kerneltype==4
    struct.sigma=canshu(1)
    struct.degree=canshu(2)
elseif  i_kerneltype==5
    struct.constant=canshu(1)
elseif  i_kerneltype==6
    struct.constant=canshu(1)
else
    struct.constant=canshu(1)
end    %Particle dimension (number of coefficients in the optimization equation) = number of parameters being optimized iteratively.


K = kernel_matrix(X_datanorm, va_kerneltype{2}, struct);

% struct.degree=2;
% struct.constant=1;
% struct.sigma=1;
% K = kernel_matrix(data_norm, 'rq', struct);
% 
% K = kernel_matrix(data_norm, 'polynomial',struct);

n = size(K,1);
one_mat = ones(n,n)/n;
K_centered = K - one_mat*K - K*one_mat + one_mat*K*one_mat;

% Compute eigenvalues and eigenvectors.
[V, D] = eig(K_centered);
[D_sorted, idx] = sort(diag(D),'descend');
V_sorted = V(:,idx);


% for i=1:size(data,2)
%     D_sorted(i,2)=D_sorted(i,1)/sum(D_sorted(:,1));
%     D_sorted(i,3)=sum(D_sorted(1:i,1))/sum(D_sorted(:,1));
% end

% Select the top k eigenvectors.
k_t=5
alpha_k = V_sorted(:,1:k_t);
data_kpca = K_centered * alpha_k;

X_kpca=data_kpca;

if i_julei==1
    %% Hierarchical clustering
    figure;
    %  scatter3(X(:,1),X(:,2),X(:,3),100,idx_cengci,'filled')
    plot(X_kpca(fitnessclugbesthuizong==1,1),X_kpca(fitnessclugbesthuizong==1,2),'r.','MarkerSize',12)
    hold on
    plot(X_kpca(fitnessclugbesthuizong==2,1),X_kpca(fitnessclugbesthuizong==2,2),'b.','MarkerSize',12)
    hold on
    % plot(X_kpca(fitnessclugbesthuizong==3,1),X_kpca(fitnessclugbesthuizong==3,2),'m.','MarkerSize',12)
    % hold on
    % plot(X_kpca(fitnessclugbesthuizong==4,1),X_kpca(fitnessclugbesthuizong==4,2),'g.','MarkerSize',12)
    % hold on

%     %                 plot(C_kmeans(:,1),C_kmeans(:,2),'kx',...
%     %                     'MarkerSize',15,'LineWidth',3)
    legend('Cluster 1','Cluster 2','Centroids',...
        'Location','NW')
    title 'Cluster Assignments and Centroids in 2 Groups'
    xlabel('kPC1','fontsize',12);  
    ylabel('kPC2','fontsize',12);   
    % zlabel('kPC3','fontsize',12);   
    % hidx=clusterdata(X,'maxclust',k,'distance',va_dist{1},'linkage',va_link{1});

elseif i_julei==2
    %% k-means
    figure;
    %             scatter3(X(:,1),X(:,2),X(:,3),100,idx_kmeans,'filled')
    plot(X_kpca(fitnessclugbesthuizong==1,1),X_kpca(fitnessclugbesthuizong==1,2),'r.','MarkerSize',12)
    hold on
    plot(X_kpca(fitnessclugbesthuizong==2,1),X_kpca(fitnessclugbesthuizong==2,2),'b.','MarkerSize',12)
    hold on
    plot(X_kpca(fitnessclugbesthuizong==3,1),X_kpca(fitnessclugbesthuizong==3,2),'m.','MarkerSize',12)
    hold on
    % plot(X_kpca(fitnessclugbesthuizong==4,1),X_kpca(fitnessclugbesthuizong==4,2),'g.','MarkerSize',12)
    % hold on

    % plot(C_kmeans(:,1),C_kmeans(:,2),'kx',...
    %     'MarkerSize',15,'LineWidth',3)
    legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
        'Location','NW')
    title 'Cluster Assignments and Centroids in 3 Groups'
    xlabel('kPC1','fontsize',12); 
    ylabel('kPC2','fontsize',12);   
    zlabel('kPC3','fontsize',12);   

    % [~,idx_test] = pdist2(C,Xtest,'euclidean','Smallest',1);
    % gscatter(Xtest(:,1),Xtest(:,2),idx_test,'bgm','ooo')
    % legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid', ...
    %     'Data classified to Cluster 1','Data classified to Cluster 2', ...
    %     'Data classified to Cluster 3');

elseif i_julei==3
    %% k-Medoids
    figure;
    %                 scatter3(X(:,1),X(:,2),X(:,3),100,idx_kmedoid,'filled')
    plot(X_kpca(fitnessclugbesthuizong==1,1),X_kpca(fitnessclugbesthuizong==1,2),'r.','MarkerSize',12)
    hold on
    plot(X_kpca(fitnessclugbesthuizong==2,1),X_kpca(fitnessclugbesthuizong==2,2),'b.','MarkerSize',12)
    hold on
    plot(X_kpca(fitnessclugbesthuizong==3,1),X_kpca(fitnessclugbesthuizong==3,2),'m.','MarkerSize',12)
    hold on
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

    % opts = statset('Display','iter');
    % [idx,C,sumd,d,midx,info] = kmedoids(X,4,'Distance','cityblock','Options',opts);
    % [idx,C,sumd,d,midx,info] = kmedoids(X,4,'Distance','hamming','Options',opts);
    % [idx,C,sumd,d,midx,info] = kmedoids(X,4,'Distance','hamming','Algorithm','pam');
    % [idx,C,sumd,d,midx,info] = kmedoids(X,4,'Distance','hamming','Algorithm','small');

elseif i_julei==4 
    %% GMM 
    figure;
    plot(X_kpca(fitnessclugbesthuizong==1,1),X_kpca(fitnessclugbesthuizong==1,2),'r.','MarkerSize',12)
    hold on
    plot(X_kpca(fitnessclugbesthuizong==2,1),X_kpca(fitnessclugbesthuizong==2,2),'b.','MarkerSize',12)
    hold on
    plot(X_kpca(fitnessclugbesthuizong==3,1),X_kpca(fitnessclugbesthuizong==3,2),'m.','MarkerSize',12)
    hold on
    % plot(X_kpca(fitnessclugbesthuizong==4,1),X_kpca(fitnessclugbesthuizong==4,2),'g.','MarkerSize',12)
    % hold on
    %     plot(C_kmeans(:,1),C_kmeans(:,2),'kx',...
    %         'MarkerSize',15,'LineWidth',3)
    legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
        'Location','NW')
    title 'Cluster Assignments and Centroids in 2 Groups'
    xlabel('kPC1','fontsize',12);   
    ylabel('kPC2','fontsize',12);   
    zlabel('kPC3','fontsize',12);   
else
    %% spectral clustering
    figure;
    plot(X_kpca(fitnessclugbesthuizong==1,1),X_kpca(fitnessclugbesthuizong==1,2),'r.','MarkerSize',12)
    hold on
    plot(X_kpca(fitnessclugbesthuizong==2,1),X_kpca(fitnessclugbesthuizong==2,2),'b.','MarkerSize',12)
    hold on
    plot(X_kpca(fitnessclugbesthuizong==3,1),X_kpca(fitnessclugbesthuizong==3,2),'m.','MarkerSize',12)
    hold on
    % plot(X_kpca(fitnessclugbesthuizong==4,1),X_kpca(fitnessclugbesthuizong==4,2),'g.','MarkerSize',12)
%             plot(C(:,1),C(:,2),'kx',...
%     %             'MarkerSize',15,'LineWidth',3)
    legend('Cluster 1','Cluster 2','Cluster 3',...
        'Location','NW');
    title('Cluster Assignments and Medoids');
    hold off
end
