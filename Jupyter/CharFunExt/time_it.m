% Functionality extended to run both functions and arbitrary code snippets.
% Here's the new version:

function runtime = time_it(code_input, options)
    % time_it - Measure the runtime of a function or code snippet.
    % The function is the simple equivalent of Python's %timeit.
    %
    % Usage:
    % runtime = time_it(code_input, options)
    %
    % Input:
    % - code_input: A function handle for the function to be measured or a string with code to be executed.
    % - options: A structure with optional fields:
    %   - r: Number of repetitions (default = 3)
    %   - n: Number of loops per repetition (default = 1000)
    %   - func_args: Cell array of arguments to be passed to the function (default = {})
    %
    % Output:
    % - runtime: A structure containing the timing statistics including allruns,
    %   average, best, worst, standard deviation (stdev), and coefficient of variation (cv_percent).
    %
    % 2024 (c) Hanc, Hancova (jozef.hanc@upjs.sk)
    % Ver.: 11-November-2024 

    % Check if at least one argument is provided
    if nargin < 1
        error('time_it:InsufficientInput', 'The code input must be provided as the argument.');
    end

    % Set default values for options
    if nargin < 2, options = struct(); end

    if ~isfield(options, 'r')
        options.r = 3; % Default number of repetitions
    end

    if ~isfield(options, 'n')
        options.n = 1000; % Default number of loops per repetition
    end

    if ~isfield(options, 'func_args')
        options.func_args = {}; % Default arguments for the function
    end

    % Extract options
    r = options.r;
    n = options.n;
    func_args = options.func_args;

    % Preallocate array to store the time taken for each repetition
    allruns = zeros(1, r);

    % Determine if the input is a function handle or a string of code
    if isa(code_input, 'function_handle')
        % Execute the provided function handle r times, each time executing it n times
        for i = 1:r
            tic;
            for j = 1:n
                code_input(func_args{:}); % Call the function handle with its arguments (if any)
            end
            allruns(i) = toc;
        end
    elseif ischar(code_input) || isstring(code_input)
        % Execute the provided code snippet r times, each time executing it n times
        for i = 1:r
            tic;
            for j = 1:n
                evalin('base', code_input); % Evaluate the code snippet in the base workspace
            end
            allruns(i) = toc;
        end
    else
        error('time_it:InvalidInput', 'The code input must be either a function handle or a string of code.');
    end

    % Calculate per-loop times
    per_loop_times = allruns / n;

    % Calculate summary characteristics
    mean_time = mean(per_loop_times);
    std_time = std(per_loop_times);
    best_time = min(per_loop_times);
    worst_time = max(per_loop_times);

    % the coefficient of variation (cv) in %
    if mean_time ~= 0
        cv = (std_time / mean_time) * 100;
    else
        cv = NaN; % Handle division by zero if the average is zero
    end

    % Store timing results in a structure based on per-loop times
    runtime.output = sprintf('%.2e s ± %.2e s (%.2g%% of mean) per loop (mean ± std. dev. of %d runs, %d loops each)', mean_time, std_time, cv, r, n);
    runtime.allruns = allruns;
    runtime.average = mean_time;
    runtime.best = best_time;
    runtime.worst = worst_time;
    runtime.stdev = std_time;
    runtime.cv = sprintf('%.2f%%', cv);
end

% Original code for executing strings without 'evalin':
% elseif ischar(code_input) || isstring(code_input)
%     % Execute the provided code snippet r times, each time executing it n times
%     for i = 1:r
%         tic;
%         for j = 1:n
%             eval(code_input); % Evaluate the code snippet
%         end
%         allruns(i) = toc;
%     end


