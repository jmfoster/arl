g = Game;
p0 = OptimalAgent(g.d);

f = zeros(5477, 14);
for i=1:5477
    f(i,:) = g.d.generateFeatureVector(i);
end
save('f.mat', 'f')