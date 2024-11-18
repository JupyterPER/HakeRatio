% define parameters
a = 1.5;
b = 1;

% define the PDF f_T for Hake ratio for specific parameters
f_k = @(t) exp(-(a^2 + b^2)/2) / (pi * (1 + t.^2));
q = @(t) (b + a * t) / sqrt(1 + t.^2);

% analytic version of PDF for f_T
fT = @(t) f_k(t) .* (1 + q(t) .* exp(1/2 * q(t).^2) .* ...
    arrayfun(@(q_val) integral(@(x) exp(-1/2 * x.^2), 0, q_val), q(t)));

% values of the function f_T for plot
t_values = linspace(-10, 10, 1000); % Define range for t
fT_values = arrayfun(fT, t_values);

% plot analytic function
figure;
plot(t_values, fT_values, 'r', 'LineWidth', 1.5);
xlim([xMin(1) xMax(1)]) % plot in 3 sigma range
xlabel('t');
ylabel('f_T(t)');
title('Plot of analytic PDF f_T');
grid on;
legend('analytic f_T(t)');
