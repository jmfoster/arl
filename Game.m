classdef Game < handle
    %GAME Controller
    %   Controls game by managing interaction between player1 and player2
    
    properties
       d = TicTacToe; %domain
       p0; %optimal player 
       
       %diagnostics
       diagnostics = 0;
    end
    
    methods
        
        function train(g, p1, p2, nGames)
            players = {p1 p2};
            for i=1:nGames
                %disp(strcat('Training Round Number: ', int2str(i)))
                AS = zeros(1,9);
                R = zeros(1,9);
                V = zeros(1,9);
                
                t = 1;
                gameOver = 0;
                fs = g.d.startingState();
                while(~gameOver)
                    %select afterstate
                    turn = mod(t-1,2)+1;
                    p = players{turn};
                    [as as_star r V_tilde V_tilde_star] = p.selectAfterstate(fs, 1);    %recruitment==1
                    AS(t) = as; 
                    R(t) = r;
                    V(t) = V_tilde; %or V_tilde_star if not doing SARSA
                    
                    %learn
                    if(t>2)
                       p.learn(AS(t-2), R(t-2), R(t-1), V(t-2), V(t));
                    end
                    if(g.d.isTerminalState(AS(t)))
                        p.learn(AS(t), R(t), 0, V(t), 0);
                        other_p = players{mod(turn,2)+1};
                        other_p.learn(AS(t-1), R(t-1), R(t), V(t-1), 0);
                        gameOver = 1;
                    end
                    
                    %prepare for next iteration
                    fs = AS(t);
                    t = t+1;
                end
                
                %apply cached V updates
                p1.applyCachedUpdatesV(p1.v_cache);
                p1.applyCachedUpdatesV(p2.v_cache);
                %apply cached U updates
                p1.applyCachedUpdatesU(p1.u_cache);
                p1.applyCachedUpdatesU(p2.u_cache);
                %apply cached exemplars
                p1.applyCachedExemplars(p1.cachedExemplars, p1.cachedExemplarValues);
                p1.applyCachedExemplars(p2.cachedExemplars, p2.cachedExemplarValues);
                %apply cached schemas from p1 and p2 to p1
                %apply cached schemas (also expands v, u, h representations)
   
                %p1.applyCachedSchemas(p1.schema_cache, p1.schema_cached_values);
                %p1.applyCachedSchemas(p2.schema_cache, p2.schema_cached_values);
                schema_cache = [p1.schema_cache p2.schema_cache];
                schema_cached_values = [p1.schema_cached_values p2.schema_cached_values];
                %if(length(schema_cache)>0)
                %    schema_cache
                %end
                p1.applyCachedSchemas(schema_cache, schema_cached_values);
%               %p1.applyCachedSchemas(p1.base_cache, p1.base_value_cache, p1.target_cache, p1.target_value_cache);
                %p1.applyCachedSchemas(p1.schema_cache, p1.schema_cached_values);
                %p2.nSchemasByTurn = p1.nSchemasByTurn; %need to set p2's schemasByTurn and schemasByTurnSizes here so yoked model's turns are handled correctly
                %p2.nSchemasByTurnSizes = p1.nSchemasByTurnSizes; %need to set p2's schemasByTurn and schemasByTurnSizes here so yoked model's turns are handled correctly
%               %p1.applyCachedSchemas(p2.base_cache, p2.base_value_cache, p2.target_cache, p2.target_value_cache);
                %p1.applyCachedSchemas(p2.schema_cache, p2.schema_cached_values);
                %p2.nSchemasByTurn = p1.nSchemasByTurn; %need to set p2's schemasByTurn and schemasByTurnSizes here so yoked model's turns are handled correctly
                %p2.nSchemasByTurnSizes = p1.nSchemasByTurnSizes; %need to set p2's schemasByTurn and schemasByTurnSizes here so yoked model's turns are handled correctly
                %prune exemplars
                p1.pruneExemplars;
                %merge exemplar sets
                %p1.mergeExemplarSet(p2.E, p2.v); replaced this step by caching exemplars until end of game
                p2.E = p1.E;
                p2.h = p1.h;    %note: p2's h is forever linked to p1's, but should be okay for self-play [or is it passed byval?]
                p2.v = p1.v;    %p2's v is forever linked to p1's v, but should be okay for self-play [or is it passed byval?]
                p2.u = p1.u;
                p2.z = p1.z;
                p2.nValues = p1.nValues; %update p2's nValues so that its v_cache gets reset with length for the correct number of states+schemas
                %reset cached v updates back to 0
                p1.reset_v_cache;
                p2.reset_v_cache; 
                %reset cached u updates back to 0
                p1.reset_u_cache;
                p2.reset_u_cache; 
                %p2.v_cache = p1.v_cache; %important that v_cache is passed byval, not byref
                %reset cached schemas
                p1.reset_schema_caches;
                p2.reset_schema_caches;
                %reset cached Exemplars
                p1.resetCachedExemplars;
                p2.resetCachedExemplars;

                %diagnostics
                if(g.diagnostics>1)
                    ag = p1;
                    gen = zeros(length(ag.E),1);
                    for e = 1:length(ag.E)
                        gen(e) = ag.exemplarGeneration(ag.E(e));
                    end
                    exemplarUs = [(ag.E>5477)' (ag.E')  ag.u(ag.E) ag.v(ag.E) gen]
                    %exemplarUs = ['schema?' 'E' 'u' 'v'; exemplarUs]
                end
                
                 
            end
        end
        
        %p1 is first player, p2 is second player
        %winner = 0 for draw, 1 for p1, 2 for p2
        %points1 is for p1, points2 is for p2
        function [winner points1 points2] = play(g, p1, p2)
            players = {p1 p2};
            t = 1;
            gameOver = 0;
            points = [0 0];
            fs = g.d.startingState();
            first = 1;
            while(~gameOver)
                %select afterstate
                turn = mod(t-1,2)+1;
                p = players{turn};
                [as as_star r V_tilde V_tilde_star] = p.selectAfterstate(fs, 0); %don't recruit new exemplars during testing
                
                %store losing move for points calculation
                if(V_tilde==1 && first)  %needs to be first time V==1
                    first = 0;
                    %points for loser
                    losingMove = sum(histc(g.d.grids(as), 2))-1;
                end
                if(g.d.isTerminalState(as))
                    gameOver = 1;
                    if(r==1)
                        winner = turn;
                        %points if winner
                         points(turn) = NaN;   %points for winner
                         points(mod(turn,2)+1) = losingMove;  %points for loser
                    else
                        winner = 0;
                        %points if draw
                        points(1) = 5;
                        points(2) = 4;
                    end
                else
                    fs = as_star;    %as for softmax; as_star for greedy
                    t = t+1;
                end
            end
            points1 = points(1);
            points2 = points(2);
        end
        
        %evaluates p by playing against optimal agent with nRounds of
        %playing first as X and then as O.
        function [propWins propLosses propDraws points] = playOptimalPlayer(g, p, nRounds)
            wins = 0;
            draws = 0;
            losses = 0;
            points = 0;
            for i=1:nRounds
                %disp(strcat('Testing Round Number: ', int2str(i)))
                [g1 points1 points2] = g.play(p, g.p0);   %p goes first
                [g2 points3 points4] = g.play(g.p0, p);   %p goes second
                wins = wins + int8(g1==1) + int8(g2==2);  
                draws = draws + int8(g1==0) + int8(g2==0);
                losses = losses + int8(g1==2) + int8(g2==1);
                points = points + points1 + points4;
            end
            nGames = 2*nRounds;
            propWins = double(wins) / double(nGames);
            propDraws = double(draws) / double(nGames);
            propLosses = double(losses) / double(nGames);
        end
        
        
        function [props diffs points nSchemas] = evaluate(g, agentsA, agentsB, trainingRounds, testingRounds, nBlocks, iteration)
            props = zeros(length(agentsA), 3, nBlocks);  %model types x wins/losses/draws x blocks
            diffs = zeros(length(agentsA), nBlocks); %model types x blocks
            points = zeros(length(agentsA), nBlocks); %model types x blocks
            nSchemas = zeros(length(agentsA), nBlocks); %model types x blocks
            numExemplars = zeros(length(agentsA), nBlocks); %model types x blocks
            for i=1:nBlocks
                disp(strcat('Block: ', int2str(i), '/', int2str(nBlocks)));
                %train
                for m = 1:length(agentsA) %for each model
                    g.train(agentsA{m}, agentsB{m}, trainingRounds);
                end
                
                %test
                for m = 1:length(agentsA)   %for each model
                    [props(m,1,i) props(m,2,i) props(m,3,i) points(m, i)] = g.playOptimalPlayer(agentsA{m}, testingRounds);
                    nSchemas(m,i) = length(agentsA{m}.E(agentsA{m}.E>5477));
                    numExemplars(m,i) = length(agentsA{m}.E);
                    agentsA{m}.nSchemas = [agentsA{m}.nSchemas nSchemas(m,i)];
                    agentsA{m}.points = points(m,:);
                    agentsA{m}.numExemplars = [agentsA{m}.numExemplars numExemplars(m,i)];
                    
                    %exemplarTracking 
                    exemplarU = sum(agentsA{m}.u(agentsA{m}.E<=5477));
                    schemaU = sum(agentsA{m}.u(agentsA{m}.E>5477));
                    exemplarV = sum(abs(agentsA{m}.v(agentsA{m}.E<=5477)));
                    schemaV = sum(abs(agentsA{m}.v(agentsA{m}.E>5477)));
                    exemplarAvgU = exemplarU/numExemplars(m,i);
                    schemaAvgU = schemaU/nSchemas(m,i);
                    exemplarAvgV = exemplarV/numExemplars(m,i);
                    schemaAvgV = schemaV/nSchemas(m,i);
                    exemplarTracking = [exemplarU schemaU exemplarV schemaV exemplarAvgU schemaAvgU exemplarAvgV schemaAvgV]';
                    agentsA{m}.exemplarTracking = [agentsA{m}.exemplarTracking exemplarTracking];
                    
                    
                    if(g.diagnostics>0)
                        ag = agentsA{m};
                        gen = zeros(length(ag.E),1);
                        for e = 1:length(ag.E)
                            gen(e) = ag.exemplarGeneration(ag.E(e));
                        end
                        exemplarUs = [(ag.E>5477)' (ag.E')  ag.u(ag.E) ag.v(ag.E) gen]
                        %exemplarUs = ['schema?' 'E' 'u' 'v'; exemplarUs]
                    end
                end
                
                %props(:,:,i)
                latestPoints = points(:,max(1,i-9):i) %output points
                latestNSchemas = nSchemas(:,max(1,i-9):i) %output nSchemas
                latestNumExemplars = numExemplars(:,max(1,i-9):i) %output numExemplars
                %exemplarTracking labels [{'exemplarU'} {'schemaU'} {'exemplarV'} {'schemaV'} {'exemplarAvgU'} {'schemaAvgU'} {'exemplarAvgV'} {'schemaAvgV'}]']
                latestExemplarTracking = agentsA{3}.exemplarTracking(:,max(1,i-9):i)   %output exemplarTracking (for the specified agent, in this case 3)
               
                
                
                if(mod(i,10)==0)
                    %plotpoints
                    %figure;
                    clf,hold on;
                    %meanPoints = mean(reshape(points(:,1:i),10,i/10));
                    plot(mean(reshape(points(1,1:i),10,i/10)), 'k:');
                    if(size(points,1)>1)
                        plot(mean(reshape(points(2,1:i),10,i/10)), 'r--');
                    end
                    if(size(points,1)>2)
                        plot(mean(reshape(points(3,1:i),10,i/10)), 'b-');
                    end
                    title(strcat('Points. nExemplars = ', num2str(agentsA{1}.nExemplars), '. SchemaThreshold = ', num2str(agentsA{1}.schemaInductionThreshold), '. LearningRates = ', num2str(agentsA{1}.alpha_v), ',', num2str(agentsA{1}.alpha_u)));
                    %title('points');
                    drawnow;
                end
             
                if(mod(i,1000)==0)
                    saveStr = strcat('autosave_iter=', num2str(iteration), '_blocks=', int2str(i), '_nExemplars=', num2str(agentsA{1}.nExemplars), '_SchemaThreshold=', num2str(agentsA{1}.schemaInductionThreshold), '_normalizeActivation=',num2str(agentsA{1}.normalizeActivation), '_LearningRates=', num2str(agentsA{1}.alpha_v), '_', num2str(agentsA{1}.alpha_u),'.mat');
                    save(saveStr, 'points', 'agentsA');
                end
                %compare stored values to optimal player's
%                 for m = 1:length(agentsA)   %for each model
%                     p = agentsA{m};
%                     keys = p.v.keys;
%                     for k = 1:length(keys)
%                       diffs(m, i) = diffs(m, i) + p.v(keys{k}) - g.p0.v(keys{k});
%                     end
%                     diffs(m, i) = diffs(m, i) / length(keys);   %normalize 
%                 end

            
            end
        end
    end
    
    methods(Static)
        
        
 
        function time = main(agents, blocks)
            %% main function
            blocks = 100;
            agents = {};
            prepopulate = 0;    %0 for no prepopulation, 1 for prepopulation
            prepopSize = 100;
            recruitment = 2;   %0 for no recruitment, 1 for recruit all, 2 for probablistic recruitment
            trainingRounds = 10; %numGames is 1 x trainingRounds (but self-play so model sees each game from both sides)
            testingRounds = 1; %numGames is 2 x testingRounds
            scores = {};
            %agents = {};
            ticID = tic;
            iterations = 1;
            %parfor i=1:iterations
            for i=1:iterations
                disp(strcat('Iteration: ', int2str(i), '/', int2str(iterations)));
                g = Game;
                p0 = OptimalAgent(g.d);
                g.p0 = p0;
                %create separate Game and OptimalAgents (and Domains) for each set of models
%                 g1 = Game;
%                 p01 = OptimalAgent(g1.d);  
%                 g1.p0 = p01;
%                 g3 = Game;
%                 p03 = OptimalAgent(g3.d);
%                 g3.p0 = p03;
%                 g5 = Game;
%                 p05 = OptimalAgent(g5.d);
%                 g5.p0 = p05;
%                 g7 = Game;
%                 p07 = OptimalAgent(g7.d);
%                 g7.p0 = p07;
%                 g9 = Game;
%                 p09 = OptimalAgent(g9.d);
%                 g9.p0 = p09;

                %create featural players
                p1 = Agent(g.d, 1, recruitment, 0);
                p2 = Agent(g.d, 1, recruitment, 0);
                %create symmetric players
                p3 = Agent(g.d, 2, recruitment, 0);
                p4 = Agent(g.d, 2, recruitment, 0);
                
               % p3b = Agent(g.d, 2, recruitment, 0);
               % p4b = Agent(g.d, 2, recruitment, 0);
    %                 %p3b.alpha_u=0;
    %                 %p4b.alpha_u=0;
    %                 %create endowed relational mode
    %                 schema1 = [1 1 1; 3 3 3; 3 3 3];
    %                 schema2 = [1 3 3; 3 1 3; 3 3 1];
    %                 schema3 = [2 2 0; 3 3 3; 3 3 3];
    %                 schema4 = [2 3 3; 3 2 3; 3 3 0];
    %                 %schemas = [schema1 schema2 schema3 schema4];
    %                 s1 = p3b.d.id(schema1);
    %                 s2 = p3b.d.id(schema2);
    %                 s3 = p3b.d.id(schema3);
    %                 s4 = p3b.d.id(schema4);
    %                 s1 = p4b.d.id(schema1);
    %                 s2 = p4b.d.id(schema2);
    %                 s3 = p4b.d.id(schema3);
    %                 s4 = p4b.d.id(schema4);
    %                 schemas = [s1 s2 s3 s4];
    %                 p3b.applyCachedSchemas(schemas, zeros(1,length(schemas)),schemas,zeros(1,length(schemas)));
    %                 p4b.applyCachedSchemas(schemas, zeros(1,length(schemas)),schemas,zeros(1,length(schemas)));
    %                 
                %create endowed schema players
%                p5 = Agent(g.d, 3, recruitment, 0);
%                p6 = Agent(g.d, 3, recruitment, 0);
                %create schema induction players
                p7 = Agent(g.d, 2, recruitment, 1);
                p8 = Agent(g.d, 2, recruitment, 1);
                p7.schemaInductionThreshold = 8;
                p8.schemaInductionThreshold = 8;
                p7.macThreshold = 100;
                p8.macThreshold = 100;
%                 p9 = Agent(g3.d, 2, recruitment, 1);
%                 p10 = Agent(g3.d, 2, recruitment, 1);
%                 p9.schemaInductionThreshold = 6;
%                 p10.schemaInductionThreshold = 6;
%                 p9.macThreshold = 100;
%                 p10.macThreshold = 100;
%                 p11 = Agent(g5.d, 2, recruitment, 1);
%                 p12 = Agent(g5.d, 2, recruitment, 1);
%                 p11.schemaInductionThreshold = 6;
%                 p12.schemaInductionThreshold = 6;
%                 p11.macThreshold = 10000000;
%                 p12.macThreshold = 10000000;
                %yoked induction players
                yoked = 0;
                if(yoked==1)
                    load('yoked1000.mat');
                    p9 = Agent(g.d, 2, recruitment, 1);
                    p10 = Agent(g.d, 2, recruitment, 1);
                    p9.schemaInductionThreshold = -1;
                    p10.schemaInductionThreshold = -1;
                    %split guided model's schemas and schemaSizes vectors into even/odd vectors
                    nSchemasByTurn = p7.nSchemasByTurn;
                    nSchemasByTurnSizes = p7.nSchemasByTurnSizes;
                    p9.nSchemasYoked = nSchemasByTurn(1:2:end); %odd elements
                    p10.nSchemasYoked = nSchemasByTurn(2:2:end); %even elements
                    p9.nSchemasYokedSizes = nSchemasByTurnSizes(1:2:end); %odd elements
                    p10.nSchemasYokedSizes = nSchemasByTurnSizes(2:2:end); %even elements
                    agentsA = {p9};
                    agentsB = {p10};
                elseif(yoked==0)
                    agentsA = {p7};
                    agentsB = {p8};
                end
                agentsA = {p1, p3, p7};
                agentsB = {p2, p4, p8};
             
             
                agents{i} = agentsA;
                for m=1:length(agentsA)
                    agentsA{m}.blocks = blocks;
                    agentsB{m}.blocks = blocks;
                end
                %prepopulate if specified
                if(prepopulate==1)
                    sample = g.p0.sampleExemplars(prepopSize);
                    for m=1:length(agentsA)
                        pA = agentsA{m};
                        pB = agentsB{m};
                        pA.prepopulateExemplars(sample);
                        pB.prepopulateExemplars(sample);
                    end
                end
                %train and play models against optimal player
                [props diffs points nSchemas] = g.evaluate(agentsA, agentsB, trainingRounds, testingRounds, blocks, i);
                scores{i} = {props diffs points};
                for m=1:length(agentsA)
                    agentsA{m}.scores = scores{i};
                    agentsB{m}.blocks = scores{i};
                end
            end
            time = toc(ticID)
            if(yoked==0)
                %save('yoked1000_autosave.mat', 'p7', 'p8');
            end
            saveStr = strcat('autosave_results_blocks=', int2str(blocks), '_iterations=', int2str(iterations));
            save(saveStr, 'scores', 'agents');
            Game.analyzeScores(scores, length(agents{1}), blocks);
        end
        
        function plotPoints3P(p1, p2, p3)
            blocks = length(p1.nSchemas);
            figure;
            clf, hold on;
            blocksToPlot = blocks-mod(blocks,100);
            rows = 100;
            cols = blocksToPlot/100;
            %Game.plotPoints(ag.points, blocksToPlot, rows, cols);
            plot(mean(reshape(p1.points(1:blocksToPlot),rows,cols)), 'k:')
            plot(mean(reshape(p2.points(1:blocksToPlot),rows,cols)), 'r--')
            plot(mean(reshape(p3.points(1:blocksToPlot),rows,cols)), 'b-')
        end
        
        %% analysis
        function analyzeDynamics(ag)
            %% parameters
            exemplarPoolSize = ag.nExemplars
            schemaInductionThreshold = ag.schemaInductionThreshold
            blocks = length(ag.nSchemas);
            %time
            
            ag
            
            %% Points
            figure;
            blocksToPlot = blocks-mod(blocks,100);
            
            rows = 100;
            cols = blocksToPlot/100;
            %Game.plotPoints(ag.points, blocksToPlot, rows, cols);
            plot(mean(reshape(ag.points(1:blocksToPlot),rows,cols)), 'b-')

            %% Recruitment Analysis
            
            figure;
            %plot(ag.nRecruitedStates);
            plot(mean(reshape(ag.nRecruitedStates(1:blocksToPlot*10),rows*10,cols)));
            title('nRecruitedStates');
            
            figure;
            %plot(ag.nRecruitedSchemas);
            plot(mean(reshape(ag.nRecruitedSchemas(1:blocksToPlot*10),rows*10,cols)));
            title('nRecruitedSchemas');
            
            figure;
            %plot(ag.nSchemas);
            plot(mean(reshape(ag.nSchemas(1:blocksToPlot),rows,cols)));
            title('nSchemas');
            
            figure;
            %plot(ag.nInducedSchemas);
            plot(mean(reshape(ag.nInducedSchemas(1:blocksToPlot*10),rows*10,cols)));
            title('nInducedSchemas');
            
            
            %% Schemas Analysis
            
            Game.analyzeSchemas(ag)
           
            
            %% Exemplar Values, V, W, Generation
            gen = zeros(length(ag.E),1);
            for e = 1:length(ag.E)
                gen(e) = ag.exemplarGeneration(ag.E(e));
            end
            ag.E = sort(ag.E, 'descend');
            u = ag.u(ag.E);
            [sortedValues sortIndex] = sort(u, 'descend');          
            ag.E = ag.E(sortIndex);
            exemplarUs = [(ag.E>5477)' (ag.E')  ag.u(ag.E) ag.v(ag.E) gen]
            
            %% top 20 exemplars (based on u)
            % top 20 exemplars (based on u)
            for i=1:20
                id = ag.E(i)
                grid = ag.d.grids(id)
                if(ag.d.containsYYB(id))
                   'YYB'
               end
               if(ag.d.getReward(id)==1)
                   'MMM'
               end
                v = ag.v(id)
                u = ag.u(id)
            end
            
            
            figure;
            hist(double(cell2mat(values(ag.exemplarGeneration)))/10);
            title('exemplar generation histogram');
            figure;
            scatter(double(cell2mat(values(ag.exemplarGeneration)))/10, ag.u(ag.E));
            title('scatter: generation x u');
            
            Game.analyzeExemplars(ag)
           
            
        end
        
        function analyzeExemplars(ag)
           for i=1:length(ag.E)
               id = ag.E(i);
               grid = ag.d.grids(id);
               v = ag.v(id);
               u = ag.u(id);
               generation = ag.exemplarGeneration(id);
               if(ag.d.containsYYB(id))
                   id;
                   grid;
                   'YYB';
                   v;
                   u;
                   generation;
               end
               if(ag.d.getReward(id)==1)
                   id;
                   grid;
                   'MMM';
                   v;
                   u;
                   generation;
               end
           end
           
           eStates = ag.E(ag.E<=5477);
           eSchemas = ag.E(ag.E>5477);
           figure; scatter(ag.E,ag.v(ag.E)); title('scatter: E x v');
           figure; hist(ag.v(eStates)); title('State Values');
           figure; hist(ag.v(eSchemas)); title('Schema Values');
           
           figure; scatter(ag.E,ag.u(ag.E)); title('scatter: E x u');
           figure; hist(ag.u(eStates)); title('State Attentions');
           figure; hist(ag.u(eSchemas)); title('Schema Attentions');
           
           figure; scatter(ag.E,sum(ag.h(:,ag.E))); title('scatter: E x sum(h(:,E)');
           figure; hist(sum(ag.h(:,eStates))); title('State Similarities');
           figure; hist(sum(ag.h(:,eSchemas))); title('Schema Similarities');
           
           
           figure; scatter(ag.u(ag.E),ag.v(ag.E)); title('scatter:u x v');
           %plot(ag.E(ag.E>5477), h(ag.E>5477))
           %plot(ag.E(ag.E>6100), h(ag.E>6100))
           
            
        end
        
         function analyzeSchemas(ag)
            grids = ag.d.grids;
            schemaIDs = ag.d.nStates+1:ag.nValues;
            schemaIDs = schemaIDs(ismember(schemaIDs,ag.E));
            sizes = zeros(1,length(schemaIDs));
            values = zeros(1,length(schemaIDs));
            us = zeros(1,length(schemaIDs));
            for i = 1:length(schemaIDs)
                id = schemaIDs(i);
                grid = grids(id);
                value = ag.v(id);
                u = ag.u(id);
                us(i) = u;
                values(i) = value;
                size = sum(grid(:)~=3);
                sizes(i) = size;
                
            end
            
            if(~isempty(schemaIDs))
                %%%begin analysis
                figure
                hist(sizes, 0:9);
                title('schema sizes')
                
                figure
                hist(values);
                title('schema values')
                
                figure
                scatter(sizes, values);
                title('scatter: schema sizes x schema values');
                
                figure
                scatter(sizes, us);
                title('scatter: schema sizes x schema u');
                
                figure
                scatter(us, values);
                title('scatter: schema u x schema values');
                
                
                
                [minVal argmin] = min(values);
                [maxVal argmax] = max(values);
                minSchemaID = schemaIDs(argmin);
                minSchema = grids(int32(schemaIDs(argmin)));
                minVal;
                maxSchemaID = schemaIDs(argmax);
                maxSchema = grids(int32(schemaIDs(argmax)));
                maxVal;

                topSchemaIDs = schemaIDs(values>.5);
                topValues = values(values>.5);
                topUs = us(values>.5);
                for i = 1:length(topSchemaIDs)
                    id = topSchemaIDs(i);
                    grid = grids(id);
                    if(ag.d.containsYYB(id))
                        'YYB';
                    end
                    if(ag.d.getReward(id)==1)
                        'MMM';
                    end
                    value = topValues(i);
                end
            end
        end
        
        
        function analyzeScores(scores, players, blocks)
            iterations = length(scores);
            avgProps = zeros(players, 3, blocks);
            avgPoints = zeros(players, blocks);
            for i=1:iterations
                score = scores{i};
                props = score{1};
                points = score{3};
                avgProps = avgProps + props(:,:,1:blocks);
                avgPoints = avgPoints + points(:, 1:blocks);
            end
            avgProps = avgProps/iterations;
            avgPoints = avgPoints/iterations;
           
            %plot(1:blocks, avgPoints(1,:), 'k', 1:blocks, avgPoints(2,:), 'r', 1:blocks, avgPoints(3,:), 'b')
            %plot(1:blocks, squeeze(avgProps(1,3,:)), 'k', 1:blocks, squeeze(avgProps(2,3,:)), 'r', 1:blocks, squeeze(avgProps(3,3,:)), 'b')
            figure
            Game.plotPoints(avgPoints, blocks, 100, blocks/100)
            figure
            Game.plotProps(avgProps, blocks, 100, blocks/100)
           
        end
        
        function results()
            recruitment = 2;
            trainingRounds = 10;
            testingRounds = 1;
            g = Game;
            p0 = OptimalAgent(g.d);
            %g = Game
            g.p0 = p0;
            %create featural players
            p1 = Agent(g.d, 1, recruitment);
            p2 = Agent(g.d, 1, recruitment);
            %create symmetric players
            p3 = Agent(g.d, 2, recruitment);
            p4 = Agent(g.d, 2, recruitment);
            agentsA = {p1, p3};
            agentsB = {p2, p4};
            %evaluate
            nBlocks = 100000;
            props = zeros(length(agentsA), 3, 10000);  %model types x wins/losses/draws x blocks
            %diffs = zeros(length(agentsA), 0); %model types x blocks
            points = zeros(length(agentsA), 10000); %model types x blocks
            i = 1;
            ticID = tic;
            while i <= nBlocks
                disp(strcat('Block: ', int2str(i), '/', int2str(nBlocks)));
                %train
                for m = 1:length(agentsA) %for each model
                    g.train(agentsA{m}, agentsB{m}, trainingRounds);
                end
                %test
                for m = 1:length(agentsA)   %for each model
                   [props(m,1,i) props(m,2,i) props(m,3,i) points(m, i)] = g.playOptimalPlayer(agentsA{m}, testingRounds);
%                     props(m,1,:) = [props(m,1,:) propWins];
%                     props(m,2,:) = [props(m,2,:) propLosses];
%                     props(m,3,:) = [props(m,3,:) propDraws];
%                     points(m, :) = [points(m,:) curPoints];
                end
                props(:,:,i)
                points(:,1:i)
                i = i+1;
            end
            toc(ticID)
        end
        
        function plotProps(props, blocks, rows, cols)
            %plots
            clf,hold on
            if(size(props, 1)==3)
                plot(mean(reshape(props(3,3,1:blocks),rows,cols)), 'b', 'LineWidth', 2)
            end
            if(size(props, 1)>=2)
                 plot(mean(reshape(props(2,3,1:blocks),rows,cols)),'r','LineWidth',2)
            end
            plot(mean(reshape(props(1,3,1:blocks),rows,cols)),'k','LineWidth',2)
            title('proportion draws')
            
        end
        
        function plotPoints(points, blocks, rows, cols)
            fontSize=40;
            lineWidth = 10;
            clf,hold on
            if(size(points, 1)==3)
                plot(mean(reshape(points(3,1:blocks),rows,cols)), 'b-', 'LineWidth', lineWidth)
            end
            if(size(points,1)>=2)
                plot(mean(reshape(points(2,1:blocks),rows,cols)), 'r--', 'LineWidth', lineWidth)
            end
            plot(mean(reshape(points(1,1:blocks),rows,cols)), 'k:', 'LineWidth', lineWidth)
            set(gca,'FontSize', fontSize)
            xlabel('Training Games (Thousands)', 'FontSize', fontSize)
            ylabel('Points', 'FontSize', fontSize)
            %title('Averaged Learning Curves', 'FontSize', 30)
            if(size(points,1)==3)
                legend('Schema Induction Model', 'Relational Model', 'Featural Model', 'Location', 'Best')
            else
                legend('Relational Model', 'Featural Model')
            end
            %1:10*rows:blocks*10
        end
        
         
        function plotPoints2(points, blocks, rows, cols)
            clf,hold on
            if(size(points, 1)==3)
                plot(1:50:5000,mean(reshape(points(3,1:blocks),rows,cols)), 'b-', 'LineWidth', 4)
            end
            plot(1:50:5000,mean(reshape(points(2,1:blocks),rows,cols)), 'r--', 'LineWidth', 4)
            plot(1:50:5000,mean(reshape(points(1,1:blocks),rows,cols)), 'k:', 'LineWidth', 4)
            set(gca, 'FontSize', 14)
            xlabel('Training Games (Thousands)', 'FontSize', 18)
            ylabel('Points', 'FontSize', 18)
            title('Averaged Learning Curves', 'FontSize', 20)
            if(size(points,1)==3)
                legend('Schema Model', 'Relational Model', 'Featural Model', 'FontSize', 18)
            else
                legend('Relational Model', 'Featural Model', 'FontSize', 18)
            end
            %1:10*rows:blocks*10
        end
        
        function plotPoints3(points, blocks, rows, cols)
            clf,hold on
            if(size(points, 1)==3)
                plot(mean(reshape(points(3,1:blocks),rows,cols)), 'b-', 'LineWidth', 4)
            end
            plot(mean(reshape(points(2,1:blocks),rows,cols)), 'r--', 'LineWidth', 4)
            plot(mean(reshape(points(1,1:blocks),rows,cols)), 'k:', 'LineWidth', 4)
            set(gca,'FontSize', 30)
            xlabel('Training Games (Thousands)', 'FontSize', 26)
            ylabel('Points', 'FontSize', 18)
            %title('Averaged Learning Curves', 'FontSize', 30)
            if(size(points,1)==3)
                legend('Schema Induction Model', 'Relational Model', 'Featural Model', 'FontSize', 26)
            else
                legend('Relational Model', 'Featural Model', 'FontSize', 16)
            end
            %1:10*rows:blocks*10
        end
        
       
        
        function diagnosis()
            exemplars = p1.E;
            for j = exemplars
                grid = g.d.grids(j)
                as = g.d.id(grid);
                v = p1.v(as)
                V_tilde = p1.update_V_tilde(as)
            end
            
            keys1 = p1.v.keys
            keys2 = p2.v.keys
            values1 = p1.v.values
            values1 = cat(1, values1{:})
            values2 = p2.v.values
            values2 = cat(1, values2{:})
            max1 = find(max(values1)==values1)
            max2 = find(max(values2)==values2)
            maxas1 = keys1{max1}
            maxas2 = keys2{max2}
            maxv1 = p1.v(maxas1)
            maxv2 = p2.v(maxas2)
            grid1 = g.d.grids(maxas1)
            grid2 = g.d.grids(maxas2)
            
            min1 = find(min(values1)==values1)
            minas1 = keys1{min1}
            minv1 = p1.v(minas1)
            mingrid1 = g.d.grids(minas1)
        end
        
        function test()
            simType = 1;
            nGames = 10;
            g = Game;
            p1 = Agent(g.d, simType);
            p2 = Agent(g.d, simType);
            g.train(p1, p2, nGames);
            keys1 = p1.v.keys
            keys2 = p2.v.keys
            values1 = p1.v.values
            values1 = cat(1, values1{:})
            values2 = p2.v.values
            values2 = cat(1, values2{:})
            max1 = find(max(values1)==values1)
            max2 = find(max(values2)==values2)
            maxas1 = keys1{max1}
            maxas2 = keys2{max2}
            maxv1 = p1.v(maxas1)
            maxv2 = p2.v(maxas2)
            grid1 = g.d.grids(maxas1)
            grid2 = g.d.grids(maxas2)
            
            min1 = find(min(values1)==values1)
            minas1 = keys1{min1}
            minv1 = p1.v(minas1)
            mingrid1 = g.d.grids(minas1)
            
            exemplars = p1.E;
            for j = exemplars
                grid = g.d.grids(j)
                as = g.d.id(grid);
                v = p1.v(as)
            end
            
            %test schema player
            g = Game;
            %create schema players
            p1 = Agent(g.d, 3);
            p2 = Agent(g.d, 3);
            g.train(p1, p2, 5)
            
            p1 = p5
            p2 = p6
            keys1 = p1.v.keys
            keys2 = p2.v.keys
            values1 = p1.v.values
            values1 = cat(1, values1{:})
            values2 = p2.v.values
            values2 = cat(1, values2{:})
            max1 = find(max(values1)==values1)
            max2 = find(max(values2)==values2)
            maxas1 = keys1{max1}
            maxas2 = keys2{max2}
            maxv1 = p1.v(maxas1)
            maxv2 = p2.v(maxas2)
            grid1 = g.d.grids(maxas1)
            grid2 = g.d.grids(maxas2)
            
            s1 = p1.d.s1
            s2 = p1.d.s2
            
            grids1 = p1.d.grids
            keys = grids1.keys;
            for i=1:length(keys)
                keys{i}
                grids1(keys{i})
            end
            
       
            g = Game;
            p0 = OptimalAgent(g.d);
            %g=Game;
            g.p0 = p0;
            p1 = Agent(g.d, 1, 2);
            p2 = Agent(g.d, 1, 2);
            agentsA = {p1};
            agentsB = {p2};
            [props diffs points] = g.evaluate(agentsA, agentsB, 10, 3, 1);
            keys = p1.v.keys
            for i=1:length(keys)
                key = keys{i}
                g.d.grids(key)
                v = p1.v(key) %internal v
                V_tilde = p1.update_V_tilde(key) %external V_tilde
            end
            
            
            %prepopulation
            g = Game;
            p0 = OptimalAgent(g.d);
            %g=Game
            g.p0 = p0;

            p1 = Agent(g.d, 1);
            p2 = Agent(g.d, 1);
            p3 = Agent(g.d, 2);
            p4 = Agent(g.d, 2);
            p5 = Agent(g.d, 3);
            p6 = Agent(g.d, 3);
            agentsA = {p1, p3, p5};
            agentsB = {p2, p4, p6};
            sample = p0.sampleExemplars(10);
            for i=1:length(agentsA)
                pA = agentsA{i};
                pB = agentsB{i};
                pA.prepopulateExemplars(sample);
                pB.prepopulateExemplars(sample);
            end
            [props diffs points] = g.evaluate(agentsA, agentsB, 10, 1, 10);
            
          
            
            
         
                
        end
            
         
            
   
    end
end

