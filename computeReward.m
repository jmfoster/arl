function r = computeReward( size, g )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    r = 0;
    file = ones(size, 1);
    for i=1:size
        if(isequal(g(i, :), file'))
            r = 1;
        elseif(isequal(g(:,i), file))
            r = 1;
        elseif(isequal(diag(g), file))
            r = 1;
        elseif(isequal(diag(flipud(g)), file))
            r = 1;
        end
    end
end

