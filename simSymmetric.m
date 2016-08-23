%returns max of featural simScores between g2 and all symmetries
%of g1 (rotational and reflective)
%   midway between featural and full relational similarity for
%   tic-tac-toe domain
function [simScore bestG1 g2] = simSymmetric(g1, g2)
   %param
   theta = 1;
   %[g1 g2] = ttt.equalizePlayerPerspectives(g1, g2); don't do this
   %get all symmetries of g1
   G1 = zeros(3,3,8);
   %8 symmetries (4 rotational, 2 reflectional)
   for i=0:3
       G1(:,:,i+1) = rot90(g1, i);
   end
   g1t = g1'; %transpose for reflectional symmetries
   for i=0:3
       G1(:,:,i+5) = rot90(g1t, i);
   end
   %compute simScores
   simScores = zeros(1, 8);
   %could store similarities instead of recomputing
   for i=1:8
       g1 = G1(:,:,i);
       %need to normalize simScore for differences in size?
       comparisonIndices = g1~=3 & g2~=3;
       diff = sum(g1(comparisonIndices)~=g2(comparisonIndices));
       %normalize diff by schema size?
       %diff = diff / length(comparisonIndices);  %normalizing
       simScores(i) = exp(-theta*diff);
   end
   %return max of simScores between g2 and all symmetries of g1
   [simScore argMax] = max(simScores);

   %find best mapping
   %bestMap = find(simScores==simScore, 1);  %just take the first max
   bestG1 = G1(:,:,argMax);
   %bestAS1 = ttt.id(bestG1);
end