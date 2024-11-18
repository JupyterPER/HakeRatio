function perfStats = update_PerfStats(result, performance, display)
    % Function to update performance metrics in the structure
    % Arguments:
    % result - structure containing runtime, methodAbbr, etc.
    % performance - structure to store the arrays
    % display - optional boolean indicating whether to display the performance structure
    % 2024 (c) Hanc, Hancova (jozef.hanc@upjs.sk)
    % Ver.: 11-November-2024 

    if nargin < 3
        display = true; % Default value is true if not provided
    end

    % Extract data from result
    runtime = result.runtime;
    xN = result.xN;

    if isfield(result, 'methodAbbr')
        method = result.methodAbbr;
    end

    method = result.methodAbbr;
    
    if isfield(result, 'error')
        error = result.error;
    end
    

    % Append new values to the performance data
    if isfield(result, 'methodAbbr')
       performance.method{end+1} = method; % Store method in cell array
    end 
    performance.rt(end+1) = runtime; % Append runtime
    performance.rtPerPoint(end+1) = runtime / xN; % Append runtime per point
    performance.acceleration = performance.rt(1) ./ performance.rt; % Update acceleration

    if isfield(result, 'error')
        performance.error(end+1) = error;
    end

    % Store updated performance data
    perfStats = performance;

    
    % Display updated performance structure as a table if display is true
    if display
        performanceTable = struct2table(performance);

        % Get the field names and data from the table
        fieldNames = performanceTable.Properties.VariableNames;
        data = table2cell(performanceTable);  % Convert table to cell array for manipulation
        
        % Process data to remove {} and format with two decimal places
        for i = 1:numel(data)
            if isnumeric(data{i}) % Check if data is numeric
                data{i} = ['[', num2str(data{i}, '%.2e '), ']']; % Convert numeric values to string with two decimal places, add square brackets
            elseif iscell(data{i}) % Check if it's a cell array (e.g., 'method' field)
                data{i} = ['[', strjoin(data{i}, ', '), ']']; % Convert cell array of strings to a single string with square brackets
            end
        end
        
        % Create a new table where field names are row names
        transposedTable = table(data', 'RowNames', fieldNames, 'VariableNames', {'Performance statistics'});
        
        % Display the transposed table
       disp(transposedTable);
    end
end