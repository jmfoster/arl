classdef Agent < handle
    %AGENT for the after-state model
    %   The agent selects actions and learns.
    
    properties
        %data
        v; % =  %internal value estimates
        v_cache; %cached updates to v
        u; % exemplar-specific attention/learning rate/voting weight parameter
        u_cache; %cached updates to u
        h; % = % similarity map from (stateID, stateID) -> simScore [loaded from precomputed file]
        E = []; %exemplar set
        d; %domain object
        nValues; %number of values to store, including states and schemas if applicable
        z; %exemplar activation normalization factor
        Eact; %exemplars currently activated (retrieved) in MAC stage for current trial
        
        %cached exemplars
        cachedExemplars = [];
        cachedExemplarValues = [];
        
        %schemas
        schema_cache = []; %cached schemas induced during the game
        schema_cached_values = []; %cached value estimates for cached schemas
%         base_cache = [];
%         target_cache = [];
%         base_value_cache = [];
%         target_value_cache = [];
        
        
        %parameters
        temperature = 1; %for softmax action selection
        gamma = 1; %for temporal discounting
        alpha_v = .1; %learning rate for v
        alpha_u = .1; %learning rate for u (w)
        simType; %set by parameter passed to constructor. 
                 %1 for featural, 
                 %2 for symmetric (relational) similarity
                 %3 for symmetric+schema similarity 
        initial_v = 0;  %default value for recruited exemplars with total sim=0
                        %Note: exemplars now initialized to V_tilde
        initial_u = 1; %default value for u for initialization %w
        schemaInduction; %0 for no schema induction
                         %1 for schema induction
        schemaInductionThreshold = 2.8; %for schema induction.
                                        %-1 yoked
                                        %else, specifies z-score criteria of TD'/TD
        recruitType;     %0 no recruitment
                         %1 recruit every exemplar
                         %2 recruit probabilistically p(recruitment) = 1/#exemplars
                         %3 recruit with constant probability (1/2)
        nExemplars = 0;  %max number of exemplars model will use
                          %0 means no limit
        normalizeActivation = 0; %0 don't normalize exemplar similarity over the state space
                                 %1 normalize exemplar similarity over the state space
        limitSchemaRecruitment = 0; %0 don't limit
                                    %1 limit (probabilistically, like exemplars: p=1/#schemaExemplars)
        beginSchemaInductionTrial = 0; %prevent schema induction/recruitment from happening before training game # specified             
        allOrNoneSchemaMatches = 2; %0 for graded schema similarity
                                    %1 for all-or-none schema similarity
                                    %2 for graded decay (normalized by schema size)
        MAC = 0; %0 for no mac/fac process
                 %1 for mac/fac
        macThreshold = 10; %number of exemplars that get through MAC phase (to be FAC candidates)
        macUExp = 1; %contribution of u to mac score
        
        %diagnostics               
        dbug = 0; %0 suppress debugging information
                  %1 output debugging information
        stateVisits; %track how many times each state has been visited
        exemplarGeneration = [];
        nSchemas = [];
        numExemplars = [];
        nRecruitedSchemas = []; %number of schemas recruited into exemplar, regardless of whether they had been induced already
        nInducedSchemas = [];   %number of 'new' schemas induced that didn't exist before
        nRecruitedStates = [];
        points;
        blocks;
        scores;
        nSchemasYoked;
        nSchemasYokedSizes;
        nSchemasByTurn = [];    %number of novel schemas recruited into exemplars at each turn
        nSchemasByTurnSizes = {};  %array of schema sizes, without turn boundaries (use nSchemasByTurn to determine turn boundaries)
        turn = 0;
        sumUupdates = [];
        exemplarTracking = zeros(8, 0)
        exemplarUbyBlock = [];
        schemaUbyBlock = [];
        exemplarVbyBlock = [];
        schemaVbyBlock = [];
        
    end
    
    methods
        %constructor, initialized with reference to domain object
        function ag = Agent(domain, simType, recruitType, schemaInduction)
            ag.d = domain;
            ag.simType = simType;
            ag.recruitType = recruitType;
            ag.schemaInduction = schemaInduction;
            ag.nValues = ag.d.getMaxID();
            ag.v = zeros(ag.nValues, 1);
            ag.v_cache = zeros(ag.nValues, 1);
            ag.u = ones(ag.nValues, 1);
            ag.u_cache = zeros(ag.nValues, 1);
            ag.z = zeros(ag.nValues, 1);
            ag.stateVisits = zeros(ag.nValues, 1);
            ag.exemplarGeneration = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
    

            %the tradeoff between those thresholds makes sense.  if you want induction rate to be roughly 
            %invariant to the MAC threshold, you could try setting schemaIductionThreshold to z std deviations,
            %where z = -Phi^{-1}(k/n), where n is the mean number of exemplars that pass MAC and k is a 
            %parameter determining average number of inductions per trial, and Phi is cumulative Gaussian.
            %k = 1;
            %ag.schemaInductionThreshold = -norminv(k/ag.macthreshold);
            
            %load stored similarities according to simType
            if(simType == 1)
                ag.h = ag.d.hFeatural;
            elseif(simType == 2)
                ag.h = ag.d.hSymmetric;
            elseif(simType == 3)
                ag.h = ag.d.hSymSchema;
                ag.initializeSchemas();
            end
        end
        
        %add schemas to exemplars, and initialize their value estimates
        %schema similarity is a special case implemented in domain's
        %simSymSchema() function
        function initializeSchemas(ag)
            for s = ag.d.schemas
                ag.v = [ag.v; 0];
                ag.v_cache = [ag.v_cache; 0];
                ag.recruitExemplar(s, 0);
                ag.nValues = ag.nValues + 1;
            end
        end
        
        %possibly recruit new exemplars if recruitment==1, else don't recruit (e.g.,
        %during testing
        function [as1_soft as1_star r1 V1_tilde_soft V1_tilde_star] = selectAfterstate(ag, fs, recruitment)
            ag.turn = ag.turn+1;
            
            %get set of candidate afterstates from domain
            [AS R] = ag.d.afterstates(fs);
            
            %compute external (generalized value estimates) V_tilde for each afterstate
            V_tilde = ag.update_V_tilde(AS);
            
            %select among afterstates (softmax and greedy)
            index1_soft = ag.softargmax(V_tilde);  %softmax afterstate
            index1_star = ag.argmax(V_tilde); %greedy afterstate
            as1_soft = AS(index1_soft);   %convert from index to stateID
            as1_star = AS(index1_star);  %convert from index to stateID
            r1 = R(index1_soft);
            V1_tilde_soft = V_tilde(index1_soft); %generalized value of softmax action
            V1_tilde_star = V_tilde(index1_star); %generalized value of greedy action
            
            %increment stateVisits (for diagnostics)
            ag.stateVisits(as1_soft) = ag.stateVisits(as1_soft) + 1;

            %recruit exemplar if it's new, according to recruitment parameter
            isNewExemplar = ~ismember(as1_soft, ag.E);
            doRecruitment = recruitment==1;
            %assertions
            if(~isscalar(isNewExemplar) || ~isscalar(doRecruitment))
                dbug_as1_soft = as1_soft
                dbug_ismember = ~ismember(as1_soft, ag.E)
                dbug_recruitment = recruitment==1
                %error('isNewExemplar or doRecruitment is not scalar');
            end
            if(doRecruitment)
                if(ag.recruitType==0)   %no recruitment
                    %do nothing
                elseif(ag.recruitType==1)   %recruit every exemplar
                    ag.cacheExemplar(as1_soft, V1_tilde_soft);
                elseif(ag.recruitType==2)   %probabilistic
                    p = 1/sum(ag.E<=5477); %probability of recruitment is 1/#stateExemplars
                    if(rand<p)
                        ag.cacheExemplar(as1_soft, V1_tilde_soft);
                    end
                elseif(ag.recruitType==3)   %recruit with constant probability 1/2
                    if(rand<1/2)
                        ag.cacheExemplar(as1_soft, V1_tilde_soft); 
                    end
                end  
            end
        end
        
        %cache exemplar recruitment
        function cacheExemplar(ag, as, value)
            ag.cachedExemplars = [ag.cachedExemplars as];
            ag.cachedExemplarValues = [ag.cachedExemplarValues value];
        end
        
        %apply cached exemplars
        function applyCachedExemplars(ag, exemplars, values)
            for i = 1:length(exemplars)
                ag.recruitExemplar(exemplars(i), values(i));
            end
            ag.nRecruitedStates = [ag.nRecruitedStates length(exemplars)];
        end
        
        function resetCachedExemplars(ag)
            ag.cachedExemplars = [];
            ag.cachedExemplarValues = [];
        end
        
        %prepopulate exemplars with vector of states in sample (excluding repeats)
        function prepopulateExemplars(ag, sample)
            for s = sample
                %if(~ismember(s, ag.E))
                ag.recruitExemplar(s, 0);
                %end
            end
        end
        
        %compute generalized estimates V_tilde. if no exemplars recruited,
        %set V_tilde = 0
        function V_tilde = update_V_tilde(ag, AS)   %verify 
            V_tilde = zeros(1, length(AS)); %default value set to 0
            if(isempty(ag.E))  %no exemplars recruited
                %do nothing, V_tilde is 0's
            else
                %compute V_tilde using unique Eact for each afterstate in AS
                for i=1:length(AS)
                    as = AS(i);
                    %MAC implementation
                    if(ag.MAC==0)
                        ag.Eact = ag.E;
                    elseif(ag.MAC==1)
                        %compute mac score for each exemplar, including u
                        ag.Eact = ag.computeMACMatches(as);
                    end


                    v = ag.v(ag.Eact);
                    %if(max(ag.E)>length(ag.u))
                     %   max(ag.E)
                     %  length(ag.u)
                    %end
                    %u = exp(ag.u(ag.E));    %u is exponentiated here
                    u = ag.u(ag.Eact); %w
                    if(~ag.normalizeActivation)
                        h = ag.h(as, ag.Eact);
                    else
                        z = ag.z(ag.Eact);
                        h = repmat(z',length(as),1).*ag.h(as, ag.Eact);
                    end
                    %denom = sum(h,2); old
                    denom = h*u;
                    %V_tilde = h*v./denom; old
                    %V_tilde = h*(u.*v)./denom;
                    V_tilde(i) = h*(u.*v)./denom;
                    if(denom==0)
                        V_tilde(i) = ag.initial_v;  %if sum similarity is 0, set generalized value = initial_v (0)
                    end
                    

                    %dbug
                    %for v = V_tilde
                    %    if(isnan(v))
                    %        dbug_V_tilde = V_tilde
                    %        error('V_tilde has NaNs')
                    %    end
                    %end

                    %debug
                    if(ag.dbug)
                        for as = AS
                            simSum = 0; 
                            for j = ag.E
                                sim = ag.h(as, j);
                                simSum = simSum + sim;
                            end
                            den = sum(ag.h(as, ag.E));
                            if(den~=0 && simSum/den~=1)
                                error('normalization error in update_V_tilde()')
                            end
                        end

                        for v = V_tilde
                            if(isnan(v))
                                dbug_V_tilde = V_tilde
                                error('V_tilde has NaNs')
                            end
                        end
                    end
                end
            end
        end
        
        function macMatches = computeMACMatches(ag, AS)
            fs = ag.d.f(AS,:,:);
            fe = ag.d.f(ag.E,:,:);
            ue = ag.u(ag.E);
            

            macScores = zeros(length(AS), length(ag.E));
            for i=1:length(AS) %i is for each afterstate
                %macScores(i, :) = ag.d.hFeatural(ag, ag.E) %use featural similarity for macScore
                %macScores(i, :) = (ue.^ ag.macUExp).*dot(repmat(fs(i,:),size(fe,1),1),fe, 2) ./ (norm(fs(i,:))*sum(fe.^2,2).^.5);  %cos similiarity
                macScores(i, :) = sum(min(repmat(fs(i,:),size(fe,1),1), fe), 2)./sum(fe, 2); %overlap between s and e normalized by size of e
            end

            macActivations = sum(macScores, 1); %simply sum of dot products across all afterstates -- may be more principled way to do this
            [vals inds] = sort(macActivations, 'descend');
            macMatches = ag.E(inds(1:min(ag.macThreshold, length(ag.E))));
        end
         
        %returns index of values using Gibbs/Boltzman distribution softmax selection
        function index = softargmax(ag, values)
            if(isempty(values))
                error('passed values length == 0')
            elseif(length(values)==1)
                index = 1;
            else %Gibbs/Boltzman distribution softmax selection
                values = exp(values/ag.temperature); 
                probs = values/sum(values);
                if(sum(isnan(probs)) > 0)
                    index =  ceil(rand*length(values)); %if all values are 0
                else
                    index = find(rand(1)<=cumsum(probs),1); 
                end
            end
            
            %debug
            if(ag.dbug)
                dbug_values = values
                dbug_index = index
                if(length(index)>1)
                    index
                    error('softmax error');
                elseif(index<1 || index > length(values))
                    index
                    error('softmax error');
                elseif(~isnumeric(index))
                    index
                    error('softmax error');
                end
            end
        end
        
        %recruit exemplar j.  
        function recruitExemplar(ag, as, value)
            %n = sum(ag.E==as);  %count of this state/schema already in exemplars
            %initialize u and v
            %initializationValue = (n*ag.v(as) + value) / (n+1);    %(n*Vs + Vj) / (n+1) or (n*Vj + V_tilde) / (n+1) 
            %ag.v(as) = initializationValue;
            if(~ismember(as, ag.E))
                ag.u(as) = ag.initial_u;
                ag.v(as) = value;
                if(ag.normalizeActivation)
                    ag.z(as) = sum(ag.h(:,as));
                end
            end
            ag.E = [ag.E as];
        end
        
        %prune exemplars based on u? or v? how many to keep?
        function pruneExemplars(ag)
            allowRepeats = 0;   %param: 1 for yes, 0 for no
            ag.E = sort(ag.E, 'descend');      %sort by E as approximation to sorting by schema size as tiebreaker %verify
            if(allowRepeats)
                %do nothing
            else
                ag.E = unique(ag.E);
                %[sortedValues sortIndex] = unique(u);   %unique sorted values
                %maxIndices = sortIndex(1:min(ag.nExemplars,end));
            end
            u = ag.u(ag.E);
            if(ag.nExemplars>0) %0 means no pruning based on pool size
                [sortedValues sortIndex] = sort(u, 'descend');          %When more than one element has the same value, the order of the
                                                                        %elements are preserved in the sorted result and the indexes of
                                                                        %equal elements will be ascending in any index matrix.
                maxIndices = sortIndex(1:min(ag.nExemplars,end)); 
                ag.E = ag.E(maxIndices);    %reorders E, but shouldn't matter - their positions could change anyway by removing some
            end
            %prune exempars with negative values of u
            ag.E = ag.E(ag.u(ag.E)>=0);
            
            %diagnostics
            nextGen = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
            for e = ag.E
                if(ag.exemplarGeneration.isKey(e))
                    nextGen(e) = ag.exemplarGeneration(e)+1;
                else
                    nextGen(e) = 1;
                end
            end
            ag.exemplarGeneration = nextGen;
            
            
            %sort(ag.u(ag.E), 'descend')  %dbug
            %nSchemas = length(ag.E(ag.E>5477)) %dbug
        
        end
        
        function doSchemaInduction(ag, as1, TD, V1_tilde, denom)
            if(~isempty(ag.E))
                numer = V1_tilde*denom;
                %efficient vector implementation of TD reduction for all exemplars
                v = ag.v(ag.Eact);
                h = ag.h(as1, ag.Eact);
                u = ag.u(ag.Eact);
                numerPrimes = numer - h.*v'.*u'; %w
                denomPrimes = denom - h.*u'; %w
                V1_tildePrimes = numerPrimes ./ denomPrimes;
                delta_V1_tildes = V1_tilde - V1_tildePrimes;
                deltaTDs = delta_V1_tildes;
                if(TD==0)
                    %exemplarIDs = deltaTDs~=0;
                    exemplarIDs = logical(zeros(1,length(ag.Eact)));
                else
                    ratios = (TD+deltaTDs) ./ TD;
                    %exemplarIDs = ratios > ag.schemaInductionThreshold;
                    zscores = (ratios-mean(ratios))/std(ratios);
                    exemplarIDs = zscores > ag.schemaInductionThreshold;
                end
                if(ag.schemaInductionThreshold == -1) %yoke schema induction; might have to move this outside of ~isempty(ag.E) outer loop
                    %turn = length(ag.nSchemasByTurn)+1; %ag.nSchemasByTurn is used here to get # turns taken so far
                    turn = ag.turn;
                    nRecruits = ag.nSchemasYoked(turn); %number of schemas that should be recruited on this turn
                    if(ag.dbug && recruitSize>0)    %dbug
                        nRecruits
                    end
                    
                    %get subvector of sizes for nRecruits, located within nSchemasYokedSizes vector
                    if(nRecruits==0)
                        exemplarIDs = logical(zeros(1,length(ag.Eact))); %don't induce/recruit any schemas
                    else
                        %endPosition = sum(ag.nSchemasYoked(1:turn)); 
                        %startPosition = endPosition-nRecruits+1; 
                        %nSizes = ag.nSchemasYokedSizes(startPosition:endPosition);
                        nSizes = ag.nSchemasYokedSizes{turn};
                        sims = ag.h(as1, ag.Eact); %similarities between target and bases, used as indicator of subsequent schema size
                        sizes = ag.d.reverseSimScores(sims);
                        %recruit schemas of sizes nSizes
                        exemplarIDs = logical(zeros(1,length(ag.Eact))); %initialize exemplarIDs
                        if(length(nSizes) ~= ag.nSchemasYoked{turn}) %dbug
                            %something's wrong
                            length(nSizes)
                            ag.nSchemasYoked{turn}
                        end
                        for targetSize = nSizes
                            matches = find(sizes==targetSize);
                            if(isempty(matches)) %if no schema can be induced that would be the correct size, induce a random schema
                                match = randi(numel(exemplarIDs));
                            else
                                match = matches(randi(numel(matches))); %randomly choose from the exemplars of the correct size
                            end
                            exemplarIDs(match) = 1;
                        end
                       
                    end
                    
                    %randExemplarIDs = randperm(length(ag.E), recruitSize);
                    %%convert IDs into logical vector to be consistent with above
                    %exemplarIDs = false(1,length(ag.E));
                    %exemplarIDs(randExemplarIDs) = true;
                    
                    
                    
                end
                %cache bases and targets for later schema induction for above-threshold exemplars
                %ag.base_cache = [ag.base_cache ag.E(exemplarIDs)];
                %ag.base_value_cache = [ag.base_value_cache v(exemplarIDs)'];
                %ag.target_cache = [ag.target_cache repmat(as1, 1, sum(exemplarIDs))];
                %ag.target_value_cache = [ag.target_value_cache repmat(ag.v(as1), 1, sum(exemplarIDs))];
                bases = ag.Eact(exemplarIDs);
                baseValues = v(exemplarIDs)';
                targets = repmat(as1, 1, sum(exemplarIDs));
                targetValues = repmat(ag.v(as1), 1, sum(exemplarIDs));
                
                %ag.nSchemasByTurn = [ag.nSchemasByTurn sum(exemplarIDs==1)]; %this function is moved 
                %diagnostics
                if(ag.dbug && length(ag.E(exemplarIDs))>100)
                    TD
                    figure;
                    plot(ag.E,ratios)
                    title('v')
                    figure; plot(ag.E,u); title('u')
                    figure; plot(ag.E,h); title('h')
                    figure; plot(ag.E,numerPrimes); title('numerPrimes')
                    figure; plot(ag.E,denomPrimes); title('denomPrimes')
                    figure; plot(ag.E,ratios); title('ratios')
                    length(ag.E>5477)
                    length(ag.E)
                    ag.nSchemas
                    sum(ag.E>5477)
                    exp(-1)
                    exp(-2)
                    exp(-6)
                    h
                    h(h>.5)
                    h(h>exp(-1))
                    h(h>exp(-2))
                    plot(ag.E(ag.E>5477), h(ag.E>5477))
                    plot(ag.E(ag.E>6100), h(ag.E>6100))
                    sum(ratios>2)
                    sum(ratios>3)
                    hist(ratios)
                    sd(ratios)
                    std(ratios)
                    zscore(ratios)
                    z = zscore(ratios)
                    z
                    sum(z>3)
                    sum(z>2)
                end
                
                ag.induceAndCacheSchemas(bases, baseValues, targets, targetValues);
                
            end
        end
        
        function expandRepresentation(ag, schemas, schema_values)
             pad = 0;
             schemaList = []; %schemaList is just being used to check for duplicates, may be faster way
             for i = 1:length(schemas)
                schema = schemas(i);
                schema_value = schema_values(i);
                if(ag.nValues<schema && ~ismember(schema, schemaList)) %schema is new (not in v, v_cache, h, or nValues)
                    pad = pad+1;
                end
                schemaList = [schemaList schema];
             end
             
            %pad v, v_cache, u, u_cache, and h
            if(pad>0)
                ag.v(end+pad) = ag.initial_v;
                ag.v_cache(end+pad) = 0;
                ag.u(end+pad) = ag.initial_u;  %w
                ag.u_cache(end+pad) = 0;
                %ag.h(end+pad,end+pad) = 0;
                ag.h(:,end+pad) = 0;
                %if(size(ag.h,2)<ag.nValues+pad)
                    %pad h excessively for efficiency reasons
                 %   ag.h(:,end+100) = 0;
                %end
                ag.z(end+pad) = 0;
            end
              for i = 1:length(schemas)
                schema = schemas(i);
                schema_value = schema_values(i);
                if(ag.nValues<schema)                 
                    %update similarity h (fill pre-padded matrix of stored similarities)
                    gm = ag.d.gm;   %grid matrix
                    sg = ag.d.grids(schema);
                    match = max(squeeze(sum(sum(repmat(sg,1,1,8,5477)==gm))));  %matt's function
                    diff = 9-(match+sum(sg(:)==3));
                    %experimental
                    if(ag.allOrNoneSchemaMatches==0)
                        theta = ag.d.theta;
                        simScores = exp(-theta*diff);                      
                    elseif(ag.allOrNoneSchemaMatches==1)
                        theta = ag.d.theta;
                        simScores = exp(-theta*diff);
                        simScores(simScores<1) = 0;
                    elseif(ag.allOrNoneSchemaMatches==2)
                        schemaSize = sum(sg(:)~=3);
                        if(schemaSize==0)
                            diff = 0;
                        else
                            diff = diff/schemaSize; %divide by size of schema (# of specified grid locations)
                        end
                        theta = ag.d.theta*9;
                        simScores = exp(-theta*diff);
                    end
                    ag.h(:,schema) = simScores;
                end
              end
            ag.nValues = ag.nValues + pad;
            %diagnostics
            ag.nInducedSchemas = [ag.nInducedSchemas pad];
        end
        
        %extend v, v_cache, and E with new schemas
        function induceAndCacheSchemas(ag, bases, baseValues, targets, targetValues)
            %bases and targets aligned together, and with their respective
            %values using vector index
            
            if(ag.dbug && ~isempty(bases)) %dbug
                bases
                targets
            end
            
            dbug_nValues = ag.nValues;
            
            %build list of schemas which extend v and v_cache
            %so that h can be extended/padded in blocks
            schemas = [];
            schemaValues = [];
            for i = 1:length(bases)
                base = bases(i);
                target = targets(i);
                baseValue = baseValues(i);
                targetValue = targetValues(i);
                schema = ag.d.induceSchema(target, base);   %important to give arguments in this order!
                schema_value = baseValue;   %how to best initialize internal value of schema?
                
                %add schema to schemas and value to values
                schemas = [schemas schema];
                schemaValues = [schemaValues schema_value];
            end
            
         
            
            novelSchemaCount = 0;
            novelSchemaSizes = [];
            %turn = length(ag.nSchemasByTurnSizes)+1;
            turn = ag.turn;
            for i = 1:length(schemas)
                schema = schemas(i);
                schema_value = schemaValues(i);
%                    for as = 1:ag.nValues
%                         if(ag.simType==1)
%                             simScore = ag.d.simFeatural(as, schema);
%                         elseif(ag.simType==2)
%                             simScore = ag.d.simAnalogical(as, schema);  %old
%                         end
%                         ag.h(as, schema) = simScore;    %takes a looong time without preallocating memory
%                         ag.h(schema, as) = simScore;    %probably unnecessary because schemas will only ever be in exemplars
%                     end
      
                if(ag.limitSchemaRecruitment==0)
                     %yoking
                     if(~ismember(schema, ag.E) && ~ismember(schema, ag.schema_cache)) %if schema hasn't already been recruited before
                        novelSchemaCount = novelSchemaCount+1; 
                        novelSchemaSizes = [novelSchemaSizes ag.d.getSchemaSize(schema)];
                     end
                    
                    %ag.recruitExemplar(schema, schema_value); %recruit schema into exemplars 
                    ag.cacheSchema(schema, schema_value); 
                elseif(ag.limitSchemaRecruitment==1)
                    p = 1/sum(ag.E>=5477); %probability of recruitment is 1/#schemaExemplars
                    if(rand<p)
                        %yoking
                        if(~ismember(schema, ag.E) && ~ismember(schema, ag.schema_cache)) %if schema hasn't already been recruited before
                            novelSchemaCount = novelSchemaCount+1; 
                            novelSchemaSizes = [novelSchemaSizes ag.d.getSchemaSize(schema)];
                        end
                        
                        %ag.recruitExemplar(schema, schema_value); %recruit schema into exemplars
                        ag.cacheSchema(schema, schema_value);
                    end
                  
                end
                if(ag.dbug)
                    recruitedSchema = schema %dbug
                end
                
         
            end
            
            %yoking
            ag.nSchemasByTurn = [ag.nSchemasByTurn novelSchemaCount];
            if(isempty(schemas))
                %ag.nSchemasByTurnSizes{turn} = -1; %placeholder value
            else
                ag.nSchemasByTurnSizes{turn} = novelSchemaSizes;
            end
            

            
            %if(ag.dbug && (ag.nValues ~= dbug_nValues+pad || ag.nValues ~= length(ag.u))) %dbug
            %    ag.nValues
            %    dbug_nValues+pad
            %end

            %diagnostics
            ag.nRecruitedSchemas = [ag.nRecruitedSchemas length(schemas)];
            %ag.nInducedSchemas = [ag.nInducedSchemas pad];
            
            %dbug
            turnNum = length(ag.nSchemasByTurn);
            %if(ag.dbug && ag.nSchemasByTurn(turnNum)~=ag.nSchemasYoked(turnNum))
            %    ag.nRecruitedSchemas(turnNum)
            %    ag.nSchemasYoked(turnNum)
            %end
            
        end
        
        function cacheSchema(ag, schema, schema_value)
            ag.schema_cache = [ag.schema_cache schema];
            ag.schema_cached_values = [ag.schema_cached_values schema_value];
        end
        
        function applyCachedSchemas(ag, schema_cache, schema_cached_values)
            ag.expandRepresentation(schema_cache, schema_cached_values);
            for i=1:length(schema_cache)
                schema = schema_cache(i);
                schema_value = schema_cached_values(i);
                ag.recruitExemplar(schema, schema_value); %recruit schema into exemplars
            end          
        end
        
        %update internal estimates of value v_cache for all exemplars
        function learn(ag, as1, r1, r2, V1_tilde, V3_tilde)
            if(~isempty(ag.E))
                %debug
                if(ag.dbug==1)
                    dbug_as1 = as1
                    dbug_my_grid = ag.d.grids(as1)
                    dbug_old_v = ag.v(as1)
                end

                %compute TD error
                %TD = ag.alpha * (r1 - sqrt(ag.gamma)*r2 + ag.gamma*V3_tilde-V1_tilde); %alpha included in TD for computational efficiency (same alpha for both v and u)
                TD = (r1 - sqrt(ag.gamma)*r2 + ag.gamma*V3_tilde-V1_tilde);
                %if(TD~=0)
                 %   TD
                %end
                %denom = sum(ag.h(as1,ag.E)); %old
                %u = exp(ag.u(ag.E)); %u exponentiated here
                u = ag.u(ag.Eact); %w
                if(~ag.normalizeActivation)
                    h = ag.h(as1, ag.Eact);
                else
                    z = ag.z(ag.Eact);
                    h = z'.* ag.h(as1, ag.Eact);
                end

                v = ag.v(ag.Eact);
                denom = h*u;

                %debug
                if(ag.dbug==1 && ~isempty(denom))
                    dbug_v_delta = ag.alpha_v*(TD*h.*u' / denom)';
                    dbug_new_v = ag.v(as1) + dbug_v_delta;
                    if(TD~=0)   %debug
                        dbug_TD = TD
                    end
                    dbug_V_tilde_old = ag.update_V_tilde(as1) %debug
                end


                if(denom~=0)
                    %cache updates to v
                    %updates_v = (TD*ag.h(as1, ag.E) / denom)'; %old
                    %updates_v = (TD*h.*u' / denom)'; %u
                    updates_v = ag.alpha_v*(TD*h.*u' / denom)';  %w
                    ag.v_cache(ag.Eact) = ag.v_cache(ag.Eact) + updates_v;

                    %cache updates to u
                    %updates_u = (TD*h.*u'.*(v-V1_tilde)' / denom)';  %u   
                    updates_u = ag.alpha_u*(TD*h.*(v-V1_tilde)' / denom)';  %w
                    ag.u_cache(ag.Eact) = ag.u_cache(ag.Eact) + updates_u;
                end


                %debug
                if(ag.dbug==1)
                    if(sum(isnan(updates_v)~=0))
                        error('v_cache update(s) are NaN');
                    end
                    dbug_V_tilde_new = ag.update_V_tilde(as1) %debug
                    simSum = 0;
                    for j = ag.E
                        simSum = simSum + ag.h(as1, j);
                    end
                    if(simSum/denom ~= 1 && denom~=0)
                        debug_simSum = simSum
                        debug_denom = denom
                        debug_simSumDividedByDenom = simSum/denom
                        error('normalization error in learn()')
                    end
                end


                %initiate schema induction decision processing and caching
                if(ag.schemaInduction)
                    if(ag.beginSchemaInductionTrial <= length(ag.nRecruitedStates))    %length of ag.nRecruitedStates vector is a measure of how many training games have been played 
                        ag.doSchemaInduction(as1, TD, V1_tilde, denom)
                    else
                        ag.nRecruitedSchemas = [ag.nRecruitedSchemas 0];    %increment these vectors
                        ag.nInducedSchemas = [ag.nInducedSchemas 0];
                    end
                end
            end
        end
        
        function applyCachedUpdatesV(ag, updates_v)
            if(length(ag.v)~=length(updates_v))
                length(ag.v)
                length(updates_v)
            end  
            ag.v = ag.v + updates_v;
            ag.alpha_v = min(1/sqrt(length(ag.nSchemas)), .1);

        end
        
        function applyCachedUpdatesU(ag, updates_u)
           ag.u = ag.u + updates_u; 
           ag.sumUupdates = [ag.sumUupdates sum(updates_u)];
           ag.alpha_u = min(1/sqrt(length(ag.nSchemas)), .1);
        end
        
        function reset_u_cache(ag)
            ag.u_cache = zeros(ag.nValues, 1);
        end
        
%%% deprecated
%         function mergeExemplarSet(ag, exemplars, values)
%             for k = exemplars
%                 %if(~ismember(k, ag.E))
%                     ag.recruitExemplar(k, values(k)); 
%                 %end
%             end
%         end
        
        %reset v_cache to 0s
        function reset_v_cache(ag)
            ag.v_cache = zeros(ag.nValues, 1);
        end
        
        function reset_schema_caches(ag)
%             ag.base_cache = [];
%             ag.base_value_cache = [];
%             ag.target_cache = [];
%             ag.target_value_cache = [];
            ag.schema_cache = [];
            ag.schema_cached_values = [];
        end
    end
    
    methods(Static)
        %returns index of values using argmax selection
        function index = argmax(values)
            if(isempty(values))
                error('passed values length == 0')
            elseif(length(values)==1)
                index = 1;
            else %argmax selection
                index = find(values==max(values)); 
            end
            %if there are ties, choose from the maxes randomly
            if(length(index)>1)
               %index = RandSample(index); 
               index = index(ceil(rand*length(index)));
            end
        end
        
        function test()
%             ag = Agent(TicTacToe(), 2)
%             fs = ag.d.startingState();
%             values = [5 6 4 9 2 4 9 8 10 7 8]
%             ag.argmax(values)
%             ag.softargmax(values)
%             as1 = ag.d.worldStep(fs, 1)
%             as2 = ag.d.worldStep(as1, 5)
%             as3 = ag.d.worldStep(as2, 6)
%             as4 = ag.d.worldStep(as3, 9)
%             ag.E = [as1 as2 as3 as4]
%             [AS R] = ag.d.afterstates(as4)
            
            
            ag = Agent(TicTacToe(), 2)
            fs = ag.d.startingState()
            [as1 as1_star r1 V1_tilde V1_tilde_star] = ag.selectAfterstate(fs)
            [as2 as2_star r2 V2_tilde V2_tilde_star] = ag.selectAfterstate(as1)
            [as3 as3_star r3 V3_tilde V3_tilde_star] = ag.selectAfterstate(as2)
            ag.learn(as1, r1, r2, V1_tilde, V3_tilde, V3_tilde_star)
            
%             ag = Agent(TicTacToe(), 2)
%             fs = ag.d.startingState()
%             for i=1:9
%                 fs
%                 [as1 as1_star r1 V1_tilde V1_tilde_star] = ag.selectAfterstate(fs)
%                 ag.d.grids(as1)
%                 fs = as1;
%             end

            %test update of V_tilde
            
        end
        
    end
    
end

