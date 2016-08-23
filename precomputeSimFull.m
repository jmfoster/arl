%precompute similarity between all pairs of existing states and save in
%variables h (full matrices)

%parameters
d = TictacToe;

%create optimal agent
p0 = OptimalAgent(d);

%get set of states from optimal agent
keys = p0.v.keys;
states = double(cell2mat(keys));
size = length(states);

%create sparse matrixes
%s = zeros(size, 1);
hFeatural = zeros(size, size);
hSymmetric = zeros(size, size);
hSymSchema = zeros(size, size);

%fill in matrices with similarities
ticID = tic;
for i=1:size
   disp(strcat('i:', int2str(i), '/', int2str(size)))
   for j=i:size
       %disp(strcat('i:', int2str(i), '/', int2str(size), ', j:', int2str(j), '/',int2str(size)));
       sim1 = d.simFeaturalIDs(i, j);
       sim2 = d.simSymmetric(i, j);
       sim3 = d.simSymSchema(i, j);
       
       hFeatural(i, j) = sim1;
       hFeatural(j, i) = sim1;
       
       hSymmetric(i, j) = sim2;
       hSymmetric(j, i) = sim2;
       
       hSymSchema(i, j) = sim3;
       hSymSchema(j, i) = sim3;
   end
end
elapsedTime = toc(ticID)

%add in schema similarities to hSymSchema
s1 = size+1;    %MMM
s2 = size+2;    %YYB
hSymSchema = [hSymSchema, zeros(size, 2); zeros(2, size+2)];
%set self-similarity to 1
hSymSchema(s1, s1) = 1;
hSymSchema(s2, s2) = 1;
%check all states to see if they instantiate the schemas
for j = 1:size
    if(d.getReward(j)==1)
        hSymSchema(s1, j) = 1;
        hSymSchema(j, s1) = 1;
    end
    
    if(d.containsYYB(j))
        hSymSchema(s2, j) = 1;
        hSymSchema(j, s2) = 1;
    end
end



save('storedSimsFull.mat', 'hFeatural', 'hSymmetric', 'hSymSchema');