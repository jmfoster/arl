classdef OptimalAgent < Agent
    %OPTIMAL AGENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function oa = OptimalAgent(d)
            oa = oa@Agent(d, 0, 0, 0, 0);   %simType and recruitment doesn't matter, optimal agent doesn't generalize
            disp('initializing Optimal Agent')
            %uncomment next two lines and comment 3rd and 4th lines to re-create optimalAgent
            %[stack children] = oa.expandTree();  
            %oa.calculateOptimalValues(stack, children);
            load('optimalValues.mat');
            oa.v = optimalValues;
        end
        
        function [stack children] = expandTree(oa)
            fs = oa.d.startingState();
            q = java.util.LinkedList();
            
            stack = java.util.Stack();
            children = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            q.add(fs);
            while(~q.isEmpty())
                fs = q.remove();
                if(children.isKey(fs))  %this state already visited
                    %do nothing
                else
                    [AS R] = oa.d.afterstates(fs);
                    children(fs) = AS;
                    for as = AS
                        oa.v(as) = oa.d.rewards(as);
                        stack.push(as);
                        if(~oa.d.isTerminalState(as))
                            q.add(as);
                        end
                    end
                end
            end
        end
        
        function calculateOptimalValues(oa, stack, children)
            while(~stack.isEmpty())
                as = stack.pop();
                if(~oa.d.isTerminalState(as))
                    C = children(as);
                    len = length(C);
                    V = zeros(1, len);
                    for i=1:len  %for each child of as
                        V(i) = oa.v(C(i));
                    end
                    v_max = max(V);
                    oa.v(as) = -v_max;  %value of as (as an afterstate to p1) is -max value of as's aferstates (to p2)
                end
            end
        end
        
        %recruitment is not applicable to optimal agent but needed for
        %shared interface with Agent
        function [as as_star r V_tilde V_tilde_star] = selectAfterstate(oa, fs, recruitment)
            [AS R] = oa.d.afterstates(fs);
            len = length(AS);
            V = zeros(1, len);
            for i=1:len
                V(i) = oa.v(AS(i));
            end
            index = oa.argmax(V);
            as = AS(index);
            as_star = as;
            r = R(index);
            V_tilde = V(index);
            V_tilde_star = V_tilde;
        end
        
        
        function sample = sampleExemplars(oa, size)
            states = oa.v.keys;
            %states = cat(1, states{:});
            %sample = randsample(states, size)
            sample = zeros(1,size);
            for i=1:size
                r = ceil(rand*length(states));
                sample(i) = states{r};  %sample [as written, without replacement]
            end

        end
    end
    
    methods(Static)
        function test()
            d = TicTacToe;  
            oa = OptimalAgent(d);
            [stack children] = oa.expandTree();
            oa.calculateOptimalValues(stack, children);
            
            as = oa.d.startingState();
            fs = as;
            as = oa.selectAfterstate(fs)
            oa.d.grids(as)
        end
    end
    
end

