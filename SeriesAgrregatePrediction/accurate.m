function [acc precision]=accurate(result)
TP=0;
FP=0;
FN=0;
for i=1:size(result,1)
    if(result(i,1)==1)
        if(result(i,2)==1)
            TP=TP+1;
        end
    else
        if(result(i,2)==-1)
            FP=FP+1;
        else
            FN=FN+1;
        end
    end
end

precision = TP/(TP+FP);    
acc=FP/(FP+FN);
