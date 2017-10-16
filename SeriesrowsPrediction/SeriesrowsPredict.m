clear;
clc;
close all;
%% load
load('data.mat')
data=cat(2,Correct,fault);
d=size(data,2);
idx=randperm(d);
X = zeros(d,1);
for i=1:d
    newdata(i)=data(idx(i));
end

%% Feature Selection
alpha=0.7;
[ranking eigVect]=FS(newdata,alpha );

for i=1:size(eigVect,1)
    if(eigVect(i)> 0.11)
        rank(i)=ranking(i);
    end
end
for i=1:d
    newD{i}=newdata{1,i}(:,rank);
    newD{i}=cat(2,newD{i},newdata{1,i}(:,end));
end
%% Seperate
pTrain=0.7;
nTrainData=round(pTrain*d);
Train=newD(1:nTrainData);
Test=newD(nTrainData+1:end);

%% Train
for g=1:size(Train,2)
    temp=Train{1,g};
    MinX = min(temp);
    MaxX = max(temp);
    for ii = 1:size(temp,2)-1
        tempN(:,ii) = NormalF(temp(:,ii),MinX(ii),MaxX(ii));
    end
    tempnew = tempN(:,all(~isnan(tempN)));
    TrainM{g}={tempnew temp(1,end)};
    clear temp tempnew tempN;
end

%% Test
for g=1:size(Test,2)
    temp=Test{1,g};
    MinX = min(temp);
    MaxX = max(temp);
    for ii = 1:size(temp,2)-1
        tempN(:,ii)=NormalF(temp(:,ii),MinX(ii),MaxX(ii));
    end
    tempnew = tempN(:,all(~isnan(tempN)));
    TestM{g}={tempnew temp(1,end)};
    clear temp tempnew tempN;
end
%%
for i=1:size(TrainM,2)
    temp(i,:)=size(TrainM{1,i}{1,1},1);
end
a=min(temp);
clear temp;
for i=1:size(TrainM,2)
    temp(i,:)=size(TrainM{1,i}{1,1},2);
end
b=min(temp);
clear temp;

for k=1:a
    for j=1:size(TrainM,2)
        A{1,k}(j,:)=[TrainM{1,j}{1,1}(k,1:b) TrainM{1, j}{1, 2}]; 
    end
end
%% Test

for i=1:size(TestM,2)
    for j=1:a
        for k=1:size(TrainM,2)
            M = pdist2((A{1,j}(k,1:b))',(TestM{1,i}{1, 1}(j,1:b))','mahalanobis');% seuclidean
            meanM{i}(k,j)=mean2(M);
        end
    end
end

%% Normalize
function xN=NormalF(x,MinX,MaxX)
xN =((x-MinX)./(MaxX-MinX));
end