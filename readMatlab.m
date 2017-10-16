clear;
clc;
close all;
%%
n=27;
d=dir('*.txt');
for k=1:length(d)
    fname=d(k).name;
    D = textread(fname,'%s','delimiter','\n','whitespace',' ');
    for i=n:size(D,1)
        C(i-26,:) = strsplit(D{i},'\t');
    end
    for l=1:size(C,1)-1
        for p=2:size(C,2)
            H{k}(l,p-1)=str2num(C{l,p});
        end
    end
    clear C;
    clear D;
end
for i=1:numel(H)
    for j=1:size(H{1,i},1)
        AA(j,1)=1;
    end
    H{1,i}=cat(2,H{1,i},AA);
    clear AA;
end
