function [ranking val]=FS(newdata,alpha)
for i=1:numel(newdata)
    A=newdata{1,i};
    data(i,:)=sum(A);
    data(i,end)=A(1,end);
    clear A;
end
% Preprocessing
s_n = data(data(:,end)==-1,:);
s_p = data(data(:,end)==1,:);
mu_sn = mean(s_n);
mu_sp = mean(s_p);

% Metric 1: Mutual Information
mi_s = [];
for i = 1:size(data(:,1:end-1),2)
    mi_s = [mi_s, muteinf(data(:,i),data(:,end))];
end

% Metric 2: class separation
sep_scores = ([mu_sp - mu_sn].^2);
st   = std(s_p).^2;
st   = st+std(s_n).^2;
f=find(st==0); %% remove ones where nothing occurs
st(f)=10000;  %% remove ones where nothing occurs
sep_scores = sep_scores ./ st;

% Building the graph
vec = abs(sep_scores(:,1:end-1) + mi_s )/2;

% Building the graph
Kernel_ij = [vec'*vec] ;

Kernel_ij = Kernel_ij - min(min( Kernel_ij ));
Kernel_ij = Kernel_ij./max(max( Kernel_ij ));

% Standard Deviation
STD = std(data(:,1:end-1),[],1);
STDMatrix = bsxfun( @max, STD, STD' );
STDMatrix = STDMatrix - min(min( STDMatrix ));
sigma_ij = STDMatrix./max(max( STDMatrix ));


Kernel=(alpha*Kernel_ij+(1-alpha)*sigma_ij); 

% Eigenvector Centrality and Ranking
opts.tol = 1e-3; 
[eigVect,~]=eigs(double(Kernel),1,'lr',opts);
[val ranking]= sort( abs(eigVect),'descend');
stem(abs(sort(eigVect)));
end

