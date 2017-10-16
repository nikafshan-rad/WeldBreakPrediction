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
l=0.09;
for z=1:5
    alpha=0.7;
    [ranking eigVect]=FS(newdata,alpha );

    for i=1:size(eigVect,1)
        if(eigVect(i)> l+0.01)
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
        tempsum=sum(tempnew,2);
        TrainS{g}={tempsum temp(1,end)};
        TrainM{g}={tempnew temp(1,end)};
        clear temp tempnew tempsum tempN;
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
        tempsum=sum(tempnew,2);
        TestS{g}={tempsum temp(1,end)};
        TestM{g}={tempnew temp(1,end)};
        clear temp tempnew tempsum tempN;
    end
    for k=1:5
        % compare seri

        for i=1:size(Test,2)
            LabalM(i,1)=TestS{1,i}{1,2};
            for j=1:size(Train,2)
                mind=min(size(TrainS{1,j}{1,1},1),size(TestS{1,i}{1,1},1));
                M = pdist2(TrainS{1,j}{1,1}(1:mind,:),TestS{1,i}{1,1}(1:mind,:),'mahalanobis');% seuclidean
                meanM(i,j)=mean2(M);
                LabalM(i,j+1)=TrainS{1,j}{1,2};
                clear M;
            end
              [a b]=min(meanM(i,:));
              result(i,:)=[LabalM(i,1) LabalM(i,b+1)];
              i
              result(i,:)
        end
        [acc(k,z) precision(k,z)]=accurate(result);
        R2{k,z}=result;
    end
end
xlswrite('acc.xlsx',acc);
xlswrite('precision.xlsx',precision);


%% Normalize
function xN=NormalF(x,MinX,MaxX)
xN =((x-MinX)./(MaxX-MinX));
end
