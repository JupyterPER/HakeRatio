function [result, Rpdf] = cf_Ratio_BKB(cf1,cf2,x,options)

% 2024 (c) Hanc, Hancova (jozef.hanc@upjs.sk)
% Ver.: 17-November-2024 

%% ALGORITHM
% the integration is given via built-in 1D integrators
% optionally using parallelization

%% CHECK THE INPUT PARAMETERS
% with simple options and only for PDF calculation
narginchk(2, 4);

if nargin < 4, options = []; end
if nargin < 3, x = []; end

if ~isfield(options, 'xN')
    options.xN = 200;
end

if ~isfield(options, 'cf2derSymbolic')
    options.cf2derSymbolic = [];
    %providing symbolic derivative of cf2 
end

if ~isfield(options, 'InfinityLimits')
    options.InfinityLimits = false;
end

if ~isfield(options, 'regularization')
    options.reg_tol = 1e-15;
end

if ~isfield(options, 'rtol')
    options.rtol = 1e-3;
end

if ~isfield(options, 'parallelCPU')
    options.parallelCPU =false;
end

if ~isfield(options, 'N')
    options.N = 2^8; % Set large N to improve the precision, e.g. N = 2^10
end

if ~isfield(options, 'SixSigmaRule')
    options.SixSigmaRule = 6;
end

if ~isfield(options, 'tolDiff')
    options.tolDiff = 1e-6;
end

if ~isfield(options, 'TuningConstant')
    options.TuningConstant = 3;
% k is a tuning numerical constant for decreasing integration step dt = (2pi/k)/range
% to improve the accuracy of numerical integration
end

if ~isfield(options, 'isPlot')
    options.isPlot = false;
end

if ~isfield(options, 'chebyPts')
    options.chebyPts = 2^7+1;
end

if ~isfield(options, 'isInterp')
    options.isInterp = false;
end

%% GET/SET the DEFAULT parameters and the OPTIONS
method            = 'BKB';
isPlot            = options.isPlot;
reg_tol           = options.reg_tol;
rtol              = options.rtol;

SixSigmaRule      = options.SixSigmaRule;
tolDiff           = options.tolDiff;
k                 = options.TuningConstant;

%FIRST, SECOND MOMENTS AND SIXSIGMA INTERVALS
% auxiliary functions for estimates of moments
cft1               = cf1(tolDiff*(1:4)');
cftRe1             = real(cft1);
cftIm1             = imag(cft1);
cft2               = cf2(tolDiff*(1:4)');
cftRe2             = real(cft2);
cftIm2             = imag(cft2);

% first moments 
xMean(1) = (8*cftIm1(1)/5 - 2*cftIm1(2)/5 + 8*cftIm1(3)/105 ...
            - 2*cftIm1(4)/280) / tolDiff;
xMean(2) = (8*cftIm2(1)/5 - 2*cftIm2(2)/5 + 8*cftIm2(3)/105 ...
            - 2*cftIm2(4)/280) / tolDiff;
% second moments
xM2(1)   = (205/72 - 16*cftRe1(1)/5 + 2*cftRe1(2)/5 ...
    - 16*cftRe1(3)/315 + 2*cftRe1(4)/560) / tolDiff^2;
xStd(1)  = sqrt(xM2(1) - xMean(1)^2);
xM2(2)   = (205/72 - 16*cftRe2(1)/5 + 2*cftRe2(2)/5 ...
    - 16*cftRe2(3)/315 + 2*cftRe2(4)/560) / tolDiff^2;
xStd(2)  = sqrt(xM2(2) - xMean(2)^2);

% sixsigma range
    xMin = xMean - SixSigmaRule * xStd; 
    xMax = xMean + SixSigmaRule * xStd; 
    range = (xMax - xMin); 


% SET OF x POINTS for PDF
if ~isempty(x)
    % If x is provided as a linspace vector
    xN = length(x);
    x1 = x(:);
    xMin(1) = min(x1);
    xMax(1) = max(x1);
else
    if options.isInterp
        %% Chebyshev points if x = [];
        xN = options.chebyPts;
        x1 = (xMax(1)-xMin(1)) * (-cos(pi*(0:(xN-1)) / ...
            (xN-1)) + 1) / 2 + xMin(1);
        x = x1;
        %% abbreviation of method
        method = [method ':CC' num2str(xN)];
    else
        %% Default values if x = [];
        xN = options.xN;
        x1 = linspace(xMin(1), xMax(1), xN);
        x = x1; 
    end
end

%INTEGRATION LIMITS

if options.InfinityLimits
    % abbreviation of method
    method = [method ':Infs'];
    % integration limits
    t1_min     = -Inf;   
    t1_max     =  Inf;  
    t2_min     =  reg_tol;  % lower bound zero adjusted by a regularization constant
    t2_max     =  Inf;
else
    % abbreviation of method
    method = [method ':SixSigma'];

    % Set integral limits for t1 and t2 for numerical integration
    N                  = options.N;
    dt                 = (2*pi/k) ./ range;
    t1                 = (0.5+(-N:N))'*dt(1);
    t2                 = (0.5+(0:N))'*dt(2);
    
    t1_min = min(t1);  % cut -infinity to the estimated lower bound of cf1 support
    t1_max = max(t1);  % cut +infinity to the estimated upper bound of cf1 support
    t2_min = reg_tol;  % lower bound zero adjusted by a regularization constant
    t2_max = max(t2);  % cut +infinity to the estimated upper bound of cf2, cf2' support

end

% Symbolic or numerical derivative of cf2 for num inversion
if ~isempty(options.cf2derSymbolic)
    % abbreviation of method
    method = [method ':SymDer'];
    % symbolic derivative from option
    cf2_der = options.cf2derSymbolic;
else
    % abbreviation of method
    method = [method ':NumDer'];
    % numerical derivative of cf2 
    % using O(h^4) Richardson extrapolation
    h = 1e-5;
    cf2_der = @(x) (4/3 * (cf2(x + h) - cf2(x - h)) / (2*h)) ...
              - (1/3 * (cf2(x + 2*h) - cf2(x - 2*h)) / (4*h));
end


%% ALGORITHM

%% CALCULATION OF RATIO PDF
if options.parallelCPU
    % abbreviation of method
    method = [method ':parCPU'];

    % start parallelization
    if isempty(gcp('nocreate'))
        parpool;
    end
    %getting active CPUs in paralleliations
    poolObj = gcp('nocreate');
    numCPUs = poolObj.NumWorkers;
    
    timeVal = tic;
    % Initialize an array to store fTx values for each x
    pdf = zeros(size(x1));

    % Loop over each x value in the array
    parfor i = 1:length(x1)
        xi = x1(i); % Current x value
        
        % Define the first inner integral, which integrates over t2 for each specific t1
        first_integral = @(t1) integral(@(t2) cf2_der(-t2 - xi * t1) ./ t2, t2_min, t2_max, ...
                              'ArrayValued', true, 'RelTol', rtol);
        
        % Define the integrand for fTx
        integrand = @(t1) cf1(t1) .* first_integral(t1);
        
        % Calculate the outer integral for fTx
        pdf(i) = (1/pi^2) * real(integral(integrand, t1_min, t1_max, 'RelTol', rtol));
    end
    Rpdf    = max(0,pdf);
    tictoc = toc(timeVal);
else
    % without parallelization
    numCPUs = 1;

    timeVal = tic;
    % Initialize an array to store fTx values for each x
    pdf = zeros(size(x1));

    % Loop over each x value in the array
    for i = 1:length(x1)
        xi = x1(i); % Current x value
        
        % Define the first inner integral, which integrates over t2 for each specific t1
        first_integral = @(t1) integral(@(t2) cf2_der(-t2 - xi * t1) ./ t2, t2_min, t2_max, ...
                              'ArrayValued', true, 'RelTol', rtol);
        
        % Define the integrand for fTx
        integrand = @(t1) cf1(t1) .* first_integral(t1);
        
        % Calculate the outer integral for fTx
        pdf(i) = (1/pi^2) * real(integral(integrand, t1_min, t1_max, 'RelTol', rtol));
    end
    Rpdf    = max(0,pdf);
    tictoc = toc(timeVal);
end

% INTERPOLANTS - Interpolation Functions
if options.isInterp
    RPDF  = @(x1new) InterpPDF(x1new,x1,Rpdf);
else
    RPDF  = [];
end

%rel tolerance in abbreviation
method = [method, ':', sprintf('%.0e', rtol)];

%% RESULT
result.Description         = 'PDF of ratio X1/X2 from the char. functions';
result.inversionMethod     = 'Broda-Kan';
if ~options.isInterp
   result.quadratureMethod = 'Built-in 1D numerical integration https://www.mathworks.com/help/matlab/ref/integral.html';
else
   result.quadratureMethod = 'Built-in 1D numerical integration with Chebyshev interpolation https://www.mathworks.com/help/matlab/ref/integral.html';
end
result.methodAbbr          = method;
result.parallelCPUs        = numCPUs;
result.cf1                 = cf1;
result.cf2                 = cf2;
result.cf2derSymbolic      = options.cf2derSymbolic;
result.xN                  = xN;  
result.x                   = x1;
result.pdf                 = Rpdf;
if options.isInterp
    result.PDF             = RPDF;
end
if ~options.InfinityLimits
result.SixSigmaRule        = SixSigmaRule;
result.xMin                = xMin;
result.xMax                = xMax;
result.N                   = N;
end
result.t1_min              = t1_min;
result.t1_max              = t1_max;
result.t2_min              = reg_tol;
result.t2_max              = t2_max;
result.rtol                = rtol;
result.options             = options;
result.runtime             = tictoc;
result.runtimePerPoint     = tictoc/xN;

%% PLOT the PDF
if length(x)==1
    isPlot = false;
end

if isPlot
    if options.isInterp
        x1 = linspace(xMin(1),xMax(1),101);
        figure
        plot(x1, RPDF(x1), 'b', 'LineWidth', 1)
        xlim([xMin(1)  xMax(1)])
        grid on
        title('Plot of Num Inv PDF for ratio')
        xlabel('x')
        ylabel('pdf')   
        legend('f(t) - Chebyshev interpolation'); 
    else
        % PDF
        figure
        plot(x1, Rpdf, 'b', 'LineWidth', 1)
        xlim([xMin(1)  xMax(1)]) 
        grid on
        title('Plot of Num Inv PDF for ratio')
        xlabel('x')
        ylabel('pdf')   
        legend('f(t)'); 
end

end
