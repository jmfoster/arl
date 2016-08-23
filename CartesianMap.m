classdef CartesianMap < handle
    %CARTESIANMAP maps from (int, int) -> value
    %  implemented using nested container.Mapsg
    
    properties
        outerMap = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    end
    
    methods
        %constructor
        function cm = CartesianMap()
            %do nothing
        end
        
        function put(cm, as, j, value)
            if(cm.outerMap.isKey(as))
                innerMap = cm.outerMap(as);
                innerMap(j) = value;    %updates innerMap inside outerMap
            else
                cm.outerMap(as) = containers.Map(j, value); %create new innerMap inside outerMap
            end
        end
        
        function value = getValue(cm, as, j)
            if(cm.exists(as, j))
                innerMap = cm.outerMap(as);
                value = innerMap(j);
            else
                error(strcat('no value stored for key (', int2str(as), ',', int2str(j),')'))
            end
        end
        
        function values = getValues(cm, as)
           if(cm.outerMap.isKey(as))
               innerMap = cm.outerMap(as);
               values = innerMap.values();
               values = cat(1, values{:});
           else
               error(strcat('no value stored for key ', as))
           end
        end
        
        function bool = exists(cm, as, j)
           bool = 0;
           if(cm.outerMap.isKey(as))
                innerMap = cm.outerMap(as);
                if(innerMap.isKey(j))
                    bool = 1;
                else
                    strcat('no value stored for second key in (', int2str(as), ',', int2str(j), ')');
                end
            else
                strcat('no value stored for first key in (', as, ',', j, ')');
            end
        end
        
    end
    
end

