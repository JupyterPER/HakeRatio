function [result, Rpdf] = cf_Ratio_BK(cf1,cf2,x,options)

% 2024 (c) Hanc, Hancova (jozef.hanc@upjs.sk)
% Ver.: 11-November-2024 

% ALGORITHM for calculating the ratio PDF
% the integration is given via fully vectorized integral sum

% OPTIONS - CHECK THE INPUT PARAMETERS
% with simple options and only for PDF calculation
narginchk(2, 4);

if nargin < 4, options = []; end
if nargin < 3, x = []; end

if ~isfield(options, 'N')
    options.N = 2^8; % Set large N to improve the precision, e.g. N = 2^10
end

if ~isfield(options, 'SixSigmaRule')
    options.SixSigmaRule = 6;
end

if ~isfield(options, 'cf2derSymbolic')
    options.cf2derSymbolic = [];
    %providing symbolic derivative of cf2 
end

if ~isfield(options, 'tolDiff')
    options.tolDiff = 1e-4;
end

if ~isfield(options, 'isPlot')
    options.isPlot = false;
end

if ~isfield(options, 'xN')
    options.xN = 200;
end

if ~isfield(options, 'cftTol')
    options.cftTol = 1e-14;       
end

if ~isfield(options, 'chebyPts')
    options.chebyPts = 2^6+1;
end

if ~isfield(options, 'isInterp')
    options.isInterp = false;
end

if ~isfield(options, 'TuningConstant')
    options.TuningConstant = 3;
%% k is a tuning numerical constant for decreasing integration step dt = (2pi/k)/range
%% to improve the accuracy of numerical integration
end

% LIMITS of the numerical integration
%% parameters from options
SixSigmaRule      = options.SixSigmaRule;
tolDiff           = options.tolDiff;
k                 = options.TuningConstant;

%% auxiliary functions for estimates of moments
cft1               = cf1(tolDiff*(1:4)');
cftRe1             = real(cft1);
cftIm1             = imag(cft1);
cft2               = cf2(tolDiff*(1:4)');
cftRe2             = real(cft2);
cftIm2             = imag(cft2);

%% first moments 
xMean(1) = (8*cftIm1(1)/5 - 2*cftIm1(2)/5 + 8*cftIm1(3)/105 ...
            - 2*cftIm1(4)/280) / tolDiff;
xMean(2) = (8*cftIm2(1)/5 - 2*cftIm2(2)/5 + 8*cftIm2(3)/105 ...
            - 2*cftIm2(4)/280) / tolDiff;
%% second moments
xM2(1)   = (205/72 - 16*cftRe1(1)/5 + 2*cftRe1(2)/5 ...
    - 16*cftRe1(3)/315 + 2*cftRe1(4)/560) / tolDiff^2;
xStd(1)  = sqrt(xM2(1) - xMean(1)^2);
xM2(2)   = (205/72 - 16*cftRe2(1)/5 + 2*cftRe2(2)/5 ...
    - 16*cftRe2(3)/315 + 2*cftRe2(4)/560) / tolDiff^2;
xStd(2)  = sqrt(xM2(2) - xMean(2)^2);

%% range
xMin = xMean - SixSigmaRule * xStd; 
xMax = xMean + SixSigmaRule * xStd; 
range = (xMax - xMin); 

%% parameters and limits for the sum
N                  = options.N;
dt                 = (2*pi/k) ./ range;
t1                 = (0.5+(-N:N))'*dt(1);
t2                 = (0.5+(0:N))'*dt(2);
c                  = dt(1) * dt(2) / (pi*pi);
[T1, T2]           = meshgrid(t1, t2);  
%%% vector of all pairs t1,t2
t                  = [T1(:) T2(:)];    

% ALGORITHM 
cftTol = options.cftTol;
isPlot = options.isPlot;
method = 'BK:TR';

% SET of x points for pdf
if isempty(x)
    if options.isInterp
        %% Chebyshev points if x = [];
        xN = options.chebyPts;
        x1 = (xMax(1)-xMin(1)) * (-cos(pi*(0:(xN-1)) / ...
            (xN-1)) + 1) / 2 + xMin(1);
        %% abbreviation of method
        method = [method ':CC' num2str(xN)];
    else 
        % Default values if x = [];
        xN = options.xN;
        x1 = linspace(xMin(1), xMax(1), xN);
    end
else
    %% If x is provided as a linspace vector
    xN = length(x);
    x1 = x(:);
    xMin(1) = min(x1);
    xMax(1) = max(x1);
end
%% Common commands for both cases
x2 = 1;
[X1, X2] = meshgrid(x1, x2);
x = [X1(:) X2(:)];

% INTEGRANDS
%% First integrand term - bivariate function cf1 divided by t2
cf1_d = @(t) cf1(t(:,1)) ./ t(:,2);

%% Second integrand term - derivative of cf2
if ~isempty(options.cf2derSymbolic)
    %%% abbreviation of method
    method = [method ':SymDer'];
    %%% getting symbolic derivative from options
    cf2_der = options.cf2derSymbolic;
else
    %%% abbreviation of method
    method = [method ':NumDer'];
    %%% numerical derivative of cf2 
    %%% using O(h^4) Richardson extrapolation
    h = 1e-5;
    cf2_der = @(x) (4/3 * (cf2(x + h) - cf2(x - h)) / (2*h)) ...
              - (1/3 * (cf2(x + 2*h) - cf2(x - 2*h)) / (4*h));
end


% CALCULATION OF RATIO PDF
timeVal = tic;

%% Integrand term 1
CF1_d   = cf1_d(t);
%% optimization of the term
id      = abs(CF1_d) > cftTol;
CF1_d   = CF1_d(id);
t       = t(id,:);

%% Integrand term 2
CF2_dif = cf2_der(-x*t');

%% Ratio pdf 
pdf     = c*real(CF2_dif*CF1_d);
Rpdf    = max(0,pdf);

tictoc = toc(timeVal);

% INTERPOLANTS - Interpolation Functions
if options.isInterp
    RPDF  = @(x1new) InterpPDF(x1new,x1,Rpdf);
else
    RPDF  = [];
end

%RESULTS
result.Description         = 'PDF of ratio X1/X2 from the char. functions';
result.inversionMethod     = 'Broda-Kan';
if ~options.isInterp
   result.quadratureMethod = 'Trapezoidal 2D quadrature';
else
   result.quadratureMethod = 'Clenshaw-Curtis 2D quadrature';
end
result.methodAbbr          = method;
result.cf1                 = cf1;
result.cf2                 = cf2;
result.cf2derSymbolic      = options.cf2derSymbolic;
result.xN                  = xN;  
result.x                   = x(:,1);
result.pdf                 = Rpdf;
if options.isInterp
    result.PDF             = RPDF;
end
result.SixSigmaRule        = options.SixSigmaRule;
result.N                   = N;
result.dt                  = dt;
result.T                   = t(end);
result.xMean               = xMean;
result.xStd                = xStd;
result.xMin                = xMin;
result.xMax                = xMax;
result.options             = options;
result.runtime             = tictoc;
result.runtimePerPoint     = tictoc/xN;

% PLOT the PDF
if length(x)==1
    isPlot = false;
end

if isPlot
    % PDF
    
    if options.isInterp
        x1 = linspace(xMin(1),xMax(1),101);
        figure
        plot(x1, RPDF(x1), 'b', 'LineWidth', 1)
        xlim([xMin(1)  xMax(1)]) % plot in sigma range
        grid on
        title('Plot of Num Inv PDF for ratio')
        xlabel('x')
        ylabel('pdf')   
        legend('f(t) - Chebyshev interpolation'); 
    else 
        figure
        plot(x1, Rpdf, 'b', 'LineWidth', 1)
        xlim([xMin(1)  xMax(1)]) % plot in sigma range
        grid on
        title('Plot of Num Inv PDF for ratio')
        xlabel('x')
        ylabel('pdf')   
        legend('f(t)'); 
end

end
