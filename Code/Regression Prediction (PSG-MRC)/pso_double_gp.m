function [best_mse, best_ma, best_sigma, best_ell, best_r2, best_r2_cv, best_r2_test] = pso_double_gp(x, y, x_test,y_test)
% PSO
D=3
c1 = 2; 
c2 = 2;

maxgeneration=100;   
si=40;   

%% Adjust the bounds (parameter bounds for coefficients)
zmax=3; 
zmin=1; 
vmax=zmax/10; 
vmin=zmin/10; 
rmax=1; 
rmin=0; 
w=rand;
% load data.mat data
% i = 1;
for i=1:si

    % positive_random_number = xmin + (xmax-xmin) * rand(1,D);

    z(i,:)=zmin + (zmax-zmin) * rand(1,D);    
    v(i,:)=vmin + (vmax-vmin) * rand(1,D);  
    [fitness(i),fitnessma(i),fitnesssigma(i),fitnessell(i),fitness_r2(i),fitness_r2_cv(i), fitness_r2_test(i)]=gp_double_regression(x,y,x_test,y_test,z(i,:));   %染色体的适应度 适应度函数就是目标函数 fitness是best_sigma，最好的r2：best_r_squared
end

%% Personal best and global best

[bestfitness, bestindex]=min(fitness);
pbest=z;  
gbest=z(bestindex,:);   
fitnesspbest=fitness;   
fitnessgbest=bestfitness;   
fitnessmagbest = fitnessma(bestindex);
fitnesssigmagbest = fitnesssigma(bestindex);
fitnessellgbest = fitnessell(bestindex);
fitness_r2gbest = fitness_r2(bestindex);
fitness_r2_cvgbest = fitness_r2_cv(bestindex);
fitness_r2_testgbest = fitness_r2_test(bestindex);


%% Iterative optimization

% maxgeneration=5;  
% i = 2;
% j = 2;
% si = 3;
for i=1:maxgeneration
    i
    r=rmax-(rmax-rmin)*((i-1)/(maxgeneration-1));
    %  r=abs(rmax-(rmax-rmin)*exp(0.35)/(exp((i-1)/(maxgeneration-1))^0.35));
    %      r=rmax-(rmax-rmin)*((i-1)/(maxgeneration-1))^(1/3)
    %   r=rmax-(rmax-rmin)*exp(1-maxgeneration/i);
    w=r*sin(pi*w);
    u=mean(fitness);

    for j=1:si

        % Velocity update
        v(j,:) = w*v(j,:) + c1*rand*(pbest(j,:) - z(j,:)) + c2*rand*(gbest - z(j,:));
        v(j,find(v(j,:)>vmax))=vmax;
        v(j,find(v(j,:)<vmin))=vmin;

        % Population update
        u(i)=mean(fitness);
        w1=exp(fitness(j)/u(1))/(1+exp(-(fitness(j)/u(1))))^i;
        w2=1-w1;
        delta=w1;

        z(j,:)=w1*z(j,:)+w2*v(j,:)+rand*gbest*delta;
        z(j,find(z(j,:)>zmax))=zmax;
        z(j,find(z(j,:)<zmin))=zmin;
        %         x_xishu=x(j,:);
        %         save x_xishu;
        % Fitness value
        [fitness(j),fitnessma(j),fitnesssigma(j),fitnessell(j),fitness_r2(j),fitness_r2_cv(j), fitness_r2_test(j)]=gp_double_regression(x,y,x_test,y_test,z(j,:));   %染色体的适应度 适应度函数就是目标函数 fitness是best_sigma，最好的r2：best_r_squared

    end

    for j=1:si

        % Personal best update
        if fitness(j) < fitnesspbest(j)
            pbest(j,:) = z(j,:);
            fitnesspbest(j) = fitness(j);
            fitnessmapbest(j)=fitnessma(j);
            fitnesssigmapbest(j)=fitnesssigma(j);
            fitnessellpbest = fitnessell(j);
            fitness_r2pbest = fitness_r2(j);
            fitness_r2_cvpbest = fitness_r2_cv(j);
            fitness_r2_testpbest = fitness_r2_test(j);
        end

        % Global best update
        if fitness(j) < fitnessgbest
            gbest = z(j);
            fitnessgbest = fitness(j);
            fitnessmagbest=fitnessma(j);
            fitnesssigmagbest=fitnesssigma(j);
            fitnessellgbest = fitnessell(j);
            fitness_r2gbest = fitness_r2(j);
            fitness_r2_cvgbest = fitness_r2_cv(j);
            fitness_r2_testgbest = fitness_r2_test(j);
        end
    end
    yy(i)=fitnessgbest;
    yyy(i)=fitnessmagbest;
    yyyy(i)=fitnesssigmagbest;
    yyyyy(i)=fitnessellgbest;
    yyyyyy(i)=fitness_r2gbest;
    yyyyyyy(i)=fitness_r2_cvgbest;
    yyyyyyyy(i)=fitness_r2_testgbest;
end


gbest_shuchu=gbest;
save gbest_shuchu

%% 
fitnessgbesthuizong=fitnessgbest;% fitnessbest is the best set of coefficients, representing the optimal fitness value (i.e., the lowest MSE).
fitnessmagbesthuizong=fitnessmagbest;
fitnesssigmagbesthuizong=fitnesssigmagbest;
fitnessellhuizong=fitnessellgbest;
fitness_r2huizong=fitness_r2gbest;
fitness_r2_cvhuizong=fitness_r2_cvgbest;
fitness_r2_testhuizong=fitness_r2_testgbest;
% fitnesstestgbesthuizong=fitnesstestgbest;
    
best_mse = fitnessgbesthuizong;
best_ma = fitnessmagbesthuizong;
best_sigma = fitnesssigmagbesthuizong;
best_ell = fitnessellhuizong;
best_r2 = fitness_r2huizong;
best_r2_cv = fitness_r2_cvhuizong;
best_r2_test = fitness_r2_testhuizong;
end



