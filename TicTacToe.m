classdef TicTacToe < handle
    %TICTACTOE Domain
    %   encapsulates tic tac toe environment dynamics and grids
    %   1 == "mine"
    %   2 == "yours"
    %   Note: TicTacToe does not function properly if play continues after victory
    %   Note: schemas are currently a hack
    
    properties
        grids = containers.Map('KeyType', 'int32', 'ValueType', 'any');
        %rewards = containers.Map('KeyType', 'int32', 'ValueType', 'double');
        rewards = NaN(5477,'double'); 
        size = uint8(3); %grids will be size x size square
        %maxID;
        stateIDs;
        nStates;
        gm; %gridMatrix
        f; %feature vectors
        gridIDs;
        terminalStates;
        
        %stored similarities
        hFeatural;
        hSymmetric;
        hSymSchema;

        %schemas (hack)
        %s1 = int32(5478); %use s1 for MMM
        %s2 = int32(5479); %use s2 for YYB
        schemas;    %s1 and s2 added to this vector in TicTacToe constructor (for Agent to use)
        
        nSchemas = 0;
        
        %parameters
        theta = 1;  %used for exponential generalization in simFeaturalGrids()
    end
    
    methods
        %constructor
        function ttt = TicTacToe()
            %ttt.schemas = [ttt.s1 ttt.s2];
            %ttt.nSchemas = length(ttt.schemas);
            load('stateIDs.mat');
            ttt.stateIDs = states;
            ttt.nStates = length(ttt.stateIDs);
            %ttt.maxID = length(ttt.stateIDs);
            
            if(~exist('hFeatural', 'var') || ~exist('hSymmetric', 'var') || ~exist('hSymSchema', 'var'))
                load('storedSimsFull.mat')
            end
            ttt.hFeatural = hFeatural;
            ttt.hSymmetric = hSymmetric;
            ttt.hSymSchema = hSymSchema;
            
            load('grids.mat')
            ttt.grids = grids;
            load('rewards.mat')
            ttt.rewards = rewards;
            load('gridMatrix.mat')
            ttt.gm = gm;
            load('f.mat')
            ttt.f = f;
            load('gridIDs.mat')
            ttt.gridIDs = gridIDs;
            load('terminalStates.mat')
            ttt.terminalStates = terminalStates;
            
        end
        
        %returns starting state and adds it to grids
        function fs = startingState(ttt)
            fs = 0; %starting state ID
            grid = zeros(ttt.size, 'uint8');
            ttt.grids(fs) = grid;
        end
        
        function n = getMaxID(ttt)
            n = length(ttt.stateIDs);
        end
        
        %given grid, sets position a to 1 and return next state id (as) and
        %reward (r)
        function [as r] = worldStep(ttt, fs, a)
           if(ttt.isTerminalState(fs))
               ERROR('play continued after end of game')
               as = fs; %unreachable
               r = 0;   %unreachable
           else
               grid = ttt.grids(fs);
               %flip perspective
               %grid_new = Replace(grid, [1 2], [2 1]); %flip "mine" and "yours"
               grid_new = mod(3-grid, 3);
               turn = 1;  %it's always 'my' turn
               grid_new(a) = turn; 
               as = ttt.id(grid_new);
               ttt.grids(as) = grid_new; %add new grid to grids
               r = ttt.getReward(as); %reward automatically added to rewards
           end
        end
        
        %return reward for state id (s).  if reward not already stored in rewards map,
        %compute reward by checking for end of game
        function r = getReward(ttt, s)
            %if(ttt.rewards.isKey(s))
            %    r = ttt.rewards(s);
            if(s==0)    %starting state, special case
                r = 0;
            else
                r = ttt.rewards(s);
            end
            if(isnan(r))
                g = ttt.grids(s);
                r = computeReward_mex(ttt.size, g); %efficient c compiled code
%                 r = 0;
%                 file = ones(ttt.size, 1);
%                 for i=1:ttt.size
%                     if(isequal(g(i, :), file'))
%                         r = 1;
%                     elseif(isequal(g(:,i), file))
%                         r = 1;
%                     elseif(isequal(diag(g), file))
%                         r = 1;
%                     elseif(isequal(diag(flipud(g)), file))
%                         r = 1;
%                     end
%                 end
%                 ttt.rewards(s) = r;
               ttt.rewards(s) = r;
            end
        end
        
        %return 1 iff no actions left or reward==1
        function bool = isTerminalState(ttt, s) 
           bool = ttt.terminalStates(s+1); %have to add one because of caching including 0th starting state
                  %no actions left          or     reward==1
           %bool = isempty(ttt.getActions(s))|| ttt.getReward(s)==1;
        end
        
        %returns available actions for state s
        function actions = getActions(ttt, s)
            grid = ttt.grids(s);
            actions = find(grid(:)==0);
        end
        
        %returns vector of afterstates given fs
        function [AS R] = afterstates(ttt, fs)
            actions = ttt.getActions(fs);
            len = length(actions);
            AS = zeros(1,len);  %set of afterstates
            R = zeros(1, len); %set of rewards
            for i=1:len
               [as r] = ttt.worldStep(fs, actions(i)); 
               AS(i) = as;
               R(i) = r;
               ttt.rewards(as) = r;
            end
        end
        
        %returns analogical similarity
        function simScore = simAnalogical(ttt, as1, as2)
            %[simScore, ~, ~] = ttt.simSymmetric(as1, as2);
            [simScore, ~, ~] = simSymmetric_mex(ttt.grids(as1),ttt.grids(as2)); %SUPER efficient - compiled C code
        end
        
        %returns featural similarity given two state IDs
        function simScore = simFeaturalIDs(ttt, as1, as2)
            g1 = ttt.grids(as1);
            g2 = ttt.grids(as2);
            simScore = ttt.simFeaturalGrids(g1, g2);
        end
        
        function sizes = reverseSimScores(ttt, simScores)
            sizes = 9-(log(simScores)/-ttt.theta);
            
        end
        
        %returns max of featural simScores between g2 and all symmetries
        %of g1 (rotational and reflective)
        %   midway between featural and full relational similarity for
        %   tic-tac-toe domain
        function [simScore bestG1 g2] = simSymmetric(ttt, as1, as2)
           g1 = ttt.grids(as1);
           g2 = ttt.grids(as2);
           %[g1 g2] = ttt.equalizePlayerPerspectives(g1, g2); don't do this
           %get all symmetries of g1
           G1 = cell(1,8);
           %8 symmetries (4 rotational, 2 reflectional)
           for i=0:3
               G1{i+1} = rot90(g1, i);
           end
           g1t = g1'; %transpose for reflectional symmetries
           for i=0:3
               G1{i+5} = rot90(g1t, i);
           end
           %compute simScores
           simScores = zeros(1, 8);
           %could store similarities instead of recomputing
           for i=1:8
               simScores(i) = ttt.simFeaturalGrids(G1{i}, g2);
           end
           %return max of simScores between g2 and all symmetries of g1
           [simScore argMax] = max(simScores);
           
           %find best mapping
           %bestMap = find(simScores==simScore, 1);  %just take the first max
           bestG1 = G1{argMax};
           %bestAS1 = ttt.id(bestG1);
        end
        
        %given two grids, induce a schema that is their intersection
        %value 3 is used to indicate grid locations with mistmatched
        %values (non-intersecting grid values)
        function schemaID = induceSchema(ttt, as1, as2) 
            %find best mapping
            [simScore bestG1 g2] = ttt.simSymmetric(as1, as2);
            g1 = bestG1;
            %create schema
            schema = g2;
            schema(g1~=g2) = 3;
            schemaID = ttt.id(schema);
            %update domain rep. with new schema
            ttt.grids(schemaID) = schema;
            ttt.nSchemas = ttt.nSchemas + 1;
            %update featureVector (MAC)
            ttt.f(schemaID,:) = ttt.generateFeatureVector(schemaID);
        end
        
        function f = generateFeatureVector(ttt, as)
            grid = ttt.grids(as);
            %%features
            % #0's
            f0 = sum(grid(:)==0);
            % #1's
            f1 = sum(grid(:)==1);
            % #2's
            f2 = sum(grid(:)==2);
            % #3's
            f3 = sum(grid(:)==3);

            %compose feature vector
            f = [f0 f1 f2 f3];
            f = [f ttt.getRowTypes(as)'];
        end
        
        %returns similarity based on symmetries AND for specialized schemas
        function simScore = simSymSchema(ttt, as, j)
            if(ismember(as, ttt.schemas) && ismember(j, ttt.schemas))
                simScore = 0;   %for now, s1 and s2 have simScore==0
            elseif(as==ttt.s1)
                simScore = double(ttt.getReward(j)==1);
            elseif(as==ttt.s2)
                simScore = ttt.containsYYB(j);
            elseif(j==ttt.s1)
                simScore = double(ttt.getReward(as)==1);
            elseif(j==ttt.s2)
                simScore = ttt.containsYYB(as);
            else
                simScore = ttt.simSymmetric(as, j);
            end
        end
        
        function bool = containsYYB(ttt, s)
            g = ttt.grids(s);
            bool = 0;
            yyb = {[2 2 0]; [2 0 2]; [0 2 2]};
            for i=1:ttt.size
                for j=1:length(yyb)
                    if(isequal(g(i, :), yyb{j}))
                        bool = 1;
                    elseif(isequal(g(:,i), yyb{j}'))
                        bool = 1;
                    elseif(isequal(diag(g), yyb{j}'))
                        bool = 1;
                    elseif(isequal(diag(flipud(g)), yyb{j}'))
                        bool = 1;
                    end
                end
            end
        end
        
        
        function typeCounts = getRowTypes(ttt, s)
            g = ttt.grids(s);
       
            rowTypes = {[3 0 0]; [0 3 0]; [0 0 3]; [1 1 1]; [1 2 0]; [1 0 2]; [0 1 2]; [2 1 0]; [0 2 1]; [2 0 1]};
            typeCounts = zeros(length(rowTypes), 1);
            for j=1:length(rowTypes)
                for i=1:ttt.size
                    row = g(i,:);
                    rc = [sum(row(:)==0) sum(row(:)==1) sum(row(:)==2)];
                    if(isequal(rc, rowTypes{j}))
                        typeCounts(j) = typeCounts(j)+1;
                    end
                    row = g(:,i);
                    rc = [sum(row(:)==0) sum(row(:)==1) sum(row(:)==2)];
                   if(isequal(rc, rowTypes{j}))
                        typeCounts(j) = typeCounts(j)+1;
                    end
                end
                row = diag(g);
                rc = [sum(row(:)==0) sum(row(:)==1) sum(row(:)==2)];
                if(isequal(rc, rowTypes{j}))
                    typeCounts(j) = typeCounts(j)+1;
                end
                row = diag(flipud(g));
                rc = [sum(row(:)==0) sum(row(:)==1) sum(row(:)==2)];
                if(isequal(rc, rowTypes{j}))
                    typeCounts(j) = typeCounts(j)+1;
                end
            end
        end
        
        %returns featural similarity given two grids
        function simScore = simFeaturalGrids(ttt, g1, g2)
           %remove dummy values of schema from eligibility for comparison
           %need to normalize simScore for differences in size?
           comparisonIndices = g1~=3 & g2~=3;
           diff = sum(g1(comparisonIndices)~=g2(comparisonIndices));
           %normalize diff by schema size?
           %diff = diff / length(comparisonIndices);  %normalizing
           simScore = exp(-ttt.theta*diff);
        end
        
        %return unique int id for a grid, by converting 3x3 grid into
        %9-digit number
        function id = id(ttt, grid)
            list = grid(:)';
            if(max(list)<3) %non-schemas (base 3)
                grid = grid+1;
                id = ttt.gridIDs(grid(1), grid(2),grid(3),grid(4),grid(5),grid(6),grid(7),grid(8),grid(9));
            else
                if(max(list) < 3)  %non-schemas (base 3)
                    strn = num2str(list);
                    strn = strn(strn ~= ' ');
                    idRaw = base2dec(strn, 3)+1;
                elseif(max(list)==3)  %schemas (base 4)
                    strn = num2str(list);
                    strn = strn(strn ~= ' ');           
                    idRaw = base2dec(strn, 4)+19684; %prevent overlap b/w base3 and base4 idRaw's
                end

               id = find(ttt.stateIDs==idRaw); %squeeze id into range of 1:5477
               if(isempty(id))  %id not in stateIDs. needs to be added
                    %add idRaw to stateIDs
                    ttt.stateIDs = [ttt.stateIDs idRaw];
                    id = find(ttt.stateIDs==idRaw);
                    %error('grid not found in possible states')
               end
            end
           
        end
        
        
        function size = getSchemaSize(ttt, s)
            sg = ttt.grids(s);
            size = sum(sg(:)~=3);
        end
      
        
        function gm = createGridMatrix(ttt)
            grids = ttt.grids;
            gm = zeros(ttt.size,ttt.size,8,length(grids)-1);
            
            for j=cell2mat(grids.keys)
               if(j>0)
                   g1 = grids(j);
                   for i=0:3
                       gm(:,:,i+1,j) = rot90(g1, i);
                   end
                   g1t = g1'; %transpose for reflectional symmetries
                   for i=0:3
                       gm(:,:,i+5,j) = rot90(g1t, i);
                   end
               end
            end
        end

    end
    
    methods(Static)
        
        function gridIds = precomputeGridIDs()
            ttt = TicTacToe();
            gridIDs = zeros([3 3 3 3 3 3 3 3 3]);
            
            valueSet = values(ttt.grids);
            for i=1:length(valueSet)
                grid = valueSet{i}; 
                id = ttt.id(grid);
                grid = grid+1; %add 1 so we don't index using 0's in the grids
                %index = grid(1), grid(2),grid(3),grid(4),grid(5),grid(6),grid(7),grid(8),grid(9);
                gridIDs(grid(1), grid(2),grid(3),grid(4),grid(5),grid(6),grid(7),grid(8),grid(9)) = id;
                %gridIDs(index) = id;
                %sprintf('%1d',grid(1:9))
            end
            save('gridIDs.mat', 'gridIDs');
            
            
            
            %verify
            valueSet = values(ttt.grids);
            for i=1:length(valueSet)
                grid = valueSet{i}; 
                id1 = ttt.id(grid);
                grid = grid+1;
                id2 = gridIDs(grid(1), grid(2),grid(3),grid(4),grid(5),grid(6),grid(7),grid(8),grid(9));
                if(id1~=id2)
                    grid-1
                    id1
                    id2
                end
            end
            
        end
        
       function terminalStates = precomputeTerminalStates()
            ttt = TicTacToe();
            terminalStates = false(5478);
            
            keySet = keys(ttt.grids);
            for i=1:length(keySet)
                s = keySet{i};
                its = ttt.isTerminalState(s);
                s = s+1; %have to add one for the 0 state indexing to work
                terminalStates(s)= its;
            end
            save('terminalStates.mat', 'terminalStates');
            
            
            %verify
            keySet = keys(ttt.grids);
            for i=1:length(keySet)
                s = keySet{i};
                its = ttt.isTerminalState(s);
                
                s = s+1; %have to add one for the 0 state indexing to work
                if(its~=terminalStates(s))
                    its
                end
            end
       end
        
        

        function bool = isStartingState(s)
            bool = s==0;
        end
        
        
        %testing
        function test()
            ttt = TicTacToe()
            fs = ttt.startingState()
            [as1 r] = ttt.worldStep(fs, 1)
            ttt.isStartingState(fs)
            ttt.isStartingState(as1)
            ttt.isTerminalState(fs)
            ttt.isTerminalState(as1)
            ttt.getActions(fs)
            ttt.getActions(as1)
            [AS R] = ttt.afterstates(fs)
            [as2 r] = ttt.worldStep(as1, 3)
            [as3 r] = ttt.worldStep(as2, 5)
            ttt.grids(as1)
            ttt.grids(as2)
            ttt.grids(as3)
            [g1 g2 ] = ttt.equalizePlayerPerspectives(ttt.grids(as1), ttt.grids(as2))
            [g2 g3 ] = ttt.equalizePlayerPerspectives(ttt.grids(as2), ttt.grids(as3))
            [g1 g3 ] = ttt.equalizePlayerPerspectives(ttt.grids(as1), ttt.grids(as3))
            ttt.simSymmetric(as1, as3)
            ttt.simSymmetric(as1, as1)
            ttt.simSymmetric(as1, as2)
            
            g1 = [1 1 1; 2 2 2; 0 0 0]
            g2 = [1 2 0; 1 2 0; 1 2 0]
            id1 = ttt.id(g1)
            id2 = ttt.id(g2)
            ttt.grids(id1) = g1;
            ttt.grids(id2) = g2;
            simScore = ttt.simSymmetric(id1, id2)
            
            
            g1 = [1 1 0; 0 0 1; 0 0 0]
            g2 = [0 0 0; 0 0 1; 1 1 0]
            id1 = ttt.id(g1)
            id2 = ttt.id(g2)
            ttt.grids(id1) = g1;
            ttt.grids(id2) = g2;
            simScore = ttt.simSymmetric(id1, id2)
            
            g1 = [1 1 0; 0 0 1; 1 0 0]
            g2 = [0 1 1; 1 0 0; 0 0 1]
            id1 = ttt.id(g1)
            id2 = ttt.id(g2)
            ttt.grids(id1) = g1;
            ttt.grids(id2) = g2;
            simScore = ttt.simSymmetric(id1, id2)
            
            
            
            ttt = TicTacToe
            ag = Agent(ttt, 3)
            %g = [1 1 1; 2 2 1; 0 0 0]
            g = [1 2 0; 2 2 1; 0 2 0]
            as = ttt.id(g)
            ttt.grids(as) = g
            ag.recruitExemplar(as)
            simScore = ttt.simSymSchema(as, 19685)
            simScore = ttt.simSymSchema(as, 19686)
            
            
            
            for k = 1:length(d.grids.keys)
                g = grids{k}
                bool = 0;
                yyb = {[2 2 0]; [2 0 2]; [0 2 2]};
                for i=1:3
                    for j=1:length(yyb)
                        if(isequal(g(i, :), yyb{j}))
                            bool = 1;
                        elseif(isequal(g(:,i), yyb{j}'))
                            bool = 1;
                        elseif(isequal(diag(g), yyb{j}'))
                            bool = 1;
                        elseif(isequal(diag(flipud(g)), yyb{j}'))
                            bool = 1;
                        end
                    end
                end
                bool
            end
        end
        
        function [g1 g2] = equalizePlayerPerspectives(g1, g2)
            turn1 = histc(g1(:), 1) ~= histc(g1(:), 2) + 1;
            turn2 = histc(g2(:), 1) ~= histc(g2(:), 2) + 1;
            if(turn1 ~= turn2)
               %flip "mine" and "yours"
               %g2 = Replace(g2, [1 2], [2 1]);
               g2 = mod(3-g2, 3);
            end
        end
        
        
   
        
    end
    
end



