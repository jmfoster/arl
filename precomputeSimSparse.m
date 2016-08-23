%precompute similarity between all pairs of existing states and save in
%variables h (sparse matrices)

%parameters
d = TictacToe;

%create optimal agent
p0 = OptimalAgent(d);

%get set of states from optimal agent
keys = p0.v.keys;
states = double(cell2mat(keys));
size = length(states);

%create sparse matrixes
s = zeros(size, 1);
hFeatural = sparse(states, states, s, max(states), max(states), size^2);
hSymmetric = sparse(states, states, s, max(states), max(states), size^2);
hSymSchema = sparse(states, states, s, max(states), max(states), size^2);

ticID = tic;
%fill in matrices with similarities

for i=1:size
    disp(strcat('i:', int2str(i), '/', int2str(size)));
   for j=i:size
       %disp(strcat('i:', int2str(i), '/', int2str(size), ', j:', int2str(j), '/',int2str(size)));
       s1 = states(i);
       s2 = states(j);
       
       sim1 = d.simFeaturalIDs(s1, s2);
       sim2 = d.simSymmetric(s1, s2);
       sim3 = d.simSymSchema(s1, s2);
       
       hFeatural(s1, s2) = sim1;
       hFeatural(s2, s1) = sim1;
       
       hSymmetric(s1, s2) = sim2;
       hSymmetric(s2, s1) = sim2;
       
       hSymSchema(s1, s2) = sim3;
       hSymSchema(s2, s1) = sim3;
   end
end
elapsedTime = toc(ticID)

%get set of states from optimal agent
keys = p0.v.keys;
states = double(cell2mat(keys));
size = length(states);

%create sparse matrixes
%s = zeros(size, 1);
hFeatural = zeros(size, size);
hSymmetric = zeros(size, size);
hSymSchema = zeros(size, size);

ticID = tic;
%fill in matrices with similarities
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

save('storedSimsFull.mat', 'hFeatural', 'hSymmetric', 'hSymSchema');