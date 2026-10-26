%% HW 3 - Part 1: Intro to fluctuations
% u-velocity from a sonic anemometer, 10 Hz, midnight to 7 am, 06-Aug-2001
% Reynolds decomposition used throughout: u = bar(u) + u'

clear; clc; close all;


%% ITEM a

% load the .mat file into a struct so we can see what variables it has
S  = load('sonic_segment.mat');
fn = fieldnames(S);
disp(fn)

if numel(fn) == 1

    % case 1: everything is in one matrix (time in col 1, u in col 2)
    D = double(S.(fn{1}));

    % flip to columns if the data are stored as rows
    if size(D,2) > size(D,1)
        D = D.';
    end

    % first column is time and second column is u velocity
    t = D(:,1);
    u = D(:,2);

else

    % case 2: separate variables, loop through and determine which is which
    t = [];
    u = [];

    for k = 1:numel(fn)

        x = S.(fn{k});

        % skip anything that is not numeric data
        if ~isnumeric(x) || numel(x) < 2
            continue
        end

        % make the variable a column vector
        x = double(x(:));

        % time should continuously increase, so use that to identify it
        if isempty(t) && all(diff(x) > 0)
            t = x;

        % use the other numeric vector as the velocity
        elseif isempty(u)
            u = x;
        end
    end

    % if the file contains a variable specifically named u, use it
    if isfield(S,'u')
        u = double(S.u(:));
    end
end

% make sure both variables are column vectors
t = t(:);
u = u(:);

% number of data points
N = length(u);

% sampling rate given in the assignment
fs = 10;

% last time minus first time
span = t(end) - t(1);

% typical spacing between samples in the file's time units
dt = median(diff(t));

fprintf('\n---- ITEM a ----\n')
fprintf('N = %d points  ->  N/fs = %.0f s = %.2f hr\n', ...
    N, N/fs, N/fs/3600)
fprintf('t(end) - t(1) = %.6f\n', span)
fprintf('dt between samples = %.4e\n', dt)

% the record is about 7 hours long, so the value of span tells us the units
% about 0.29 means days, about 7 means hours, and about 25200 means seconds
if span < 1

    fprintf(['Units appear to be DAYS: %.4f days = %.2f hr, ' ...
        'dt = %.3f s\n'], span, span*24, dt*86400)

elseif span < 30

    fprintf('Units appear to be HOURS: dt = %.3f s\n', dt*3600)

else

    fprintf('Units appear to be SECONDS\n')
end

% build convenient time vectors from the known 10 Hz sampling rate
ts = (0:N-1)'/fs;                    % seconds since start
th = ts/3600;                        % hours after midnight



%% ITEM b

% choose 500 contiguous points starting about one-third through the record
% min() makes sure there are still 500 points available after the start
i0  = min(round(N/3), N-499);
idx = i0:i0+499;

% 500 points at 10 Hz represents 50 seconds of data
fprintf('\n---- ITEM b ----\n')
fprintf('500 points at 10 Hz = %.1f seconds of data\n', 500/fs)

% plot the raw signal over the selected 500-point interval
figure
plot(ts(idx) - ts(i0), u(idx), '.-')
xlabel('time from segment start [s]')
ylabel('u [m/s]')
title(sprintf('500 contiguous points starting at %.2f hr', th(i0)))
grid on



%% ITEM c

% 15 min at 10 Hz = 15*60*10 = 9000 points
n15 = 15*60*fs;

% calculate the running 15 min mean
% periodic wrapping is used at the ends, as allowed by the assignment
u15 = runmean_periodic(u, n15);

% calculate bar(u) using the entire time record
ubar = mean(u);

fprintf('\n---- ITEM c ----\n')
fprintf('15 min averaging window = %d points\n', n15)
fprintf('bar(u) over all time = %.4f m/s\n', ubar)

% plot the raw data, running 15 min mean, and overall mean together
% the running mean removes much of the rapid fluctuation so larger-scale
% changes in the mean velocity are easier to see
figure
plot(th, u, 'Color', [0.8 0.8 0.8]); hold on
plot(th, u15, 'b', 'LineWidth', 1.5)
yline(ubar, 'r--', 'LineWidth', 1.2);
xlabel('hours after midnight')
ylabel('u [m/s]')
legend('raw 10 Hz', '15 min running mean', 'bar(u) all time', ...
    'Location', 'best')
title('Running 15 min average')
grid on



%% ITEM d

% calculate fluctuations from the overall mean using u' = u - bar(u)
up = u - ubar;

% calculate bar(u'u') over the entire record
% this is the mean square of the velocity fluctuations
upup_all = mean(up.^2);

% the running calculation uses exactly 500 points = 50 seconds
n500 = 500;

% calculate the local running mean of u in each 500-point window
% periodic wrapping is used at the ends, as allowed by the assignment
um500 = runmean_periodic(u, n500);

% calculate the local running variance using
% bar(u'u') = bar(u^2) - [bar(u)]^2
upup500 = runmean_periodic(u.^2, n500) - um500.^2;

fprintf('\n---- ITEM d ----\n')
fprintf('bar(u''u'') over all time = %.4f m^2/s^2\n', upup_all)
fprintf(['mean of the 500 pt running bar(u''u'') = ' ...
    '%.4f m^2/s^2\n'], mean(upup500))

% the all-time value minus the average 500 pt value should be about equal
% to the variance of the running mean, i.e. the slow changes longer than
% 50 s that a 500 pt window can't see
fprintf('variance of the 500 pt running mean = %.4f m^2/s^2\n', ...
    var(um500,1))

% plot the running 500-point value versus time
% changes in this curve show changes in the magnitude of the short-time
% velocity fluctuations during the measurement period
figure
plot(th, upup500); hold on
yline(upup_all, 'r--', 'LineWidth', 1.2);
xlabel('hours after midnight')
ylabel('$\overline{u''u''}$ [m$^2$/s$^2$]', 'Interpreter', 'latex')
legend('500 pt running', 'all time', 'Location', 'best')
title('Running 500 pt mean of u''u''')
grid on



%% ITEM e

% choose the first 500 contiguous points for segment A
iA = 1:500;

% choose another 500 contiguous points about halfway through the record
iB = min(round(N/2), N-499) + (0:499);

% keep the original velocity values for the two segments
uA = u(iA);
uB = u(iB);

% calculate the correlation coefficient as an additional measure of how
% strongly the two selected 500-point segments are related
r = corrcoef(uA, uB);

fprintf('\n---- ITEM e ----\n')
fprintf('correlation coef between the two segments = %.3f\n', r(1,2))

% scatter one 500-point segment against the other
figure
scatter(uA, uB, 12, 'filled')
grid on
xlabel(sprintf('u, segment A (%.2f hr) [m/s]', th(iA(1))))
ylabel(sprintf('v, segment B (%.2f hr) [m/s]', th(iB(1))))
title(sprintf('Two separate 500 pt segments, r = %.3f', r(1,2)))



%% ITEM f

% averaging lengths requested in the assignment:
% 500, 1000, 10000, 100000, and all points
Mlist = [500 1000 10000 100000 N];

% create labels for the different averaging lengths
leg = cell(size(Mlist));

% integral time scale for each averaging length
Tint = zeros(size(Mlist));

fprintf('\n---- ITEM f ----\n')
fprintf('      M        R(0) [m^2/s^2]    integral time scale [s]\n')

figure

for k = 1:numel(Mlist)

    % number of points used for this autocorrelation calculation
    M = min(Mlist(k), N);

    % take the first M points and subtract their mean to obtain u'
    xp = u(1:M) - mean(u(1:M));

    % calculate lags out to half the record
    % at larger lags fewer overlapping data pairs remain
    L = floor(M/2);

    % use an FFT to efficiently calculate the autocorrelation sums
    % zero padding prevents the end of the record from wrapping onto the start
    nf = 2^nextpow2(2*M);
    F  = fft(xp, nf);
    Ssum = real(ifft(abs(F).^2));

    % lag j corresponds to Delta t = j/fs seconds
    j = (0:L)';
    lag = j/fs;

    % at lag j there are M-j overlapping pairs
    % dividing the correlation sum by M-j gives the average
    % bar[u'(t)u'(t+Delta t)] requested in the assignment
    R = Ssum(1:L+1) ./ (M-j);

    % normalize by the zero-lag value so the curve starts at 1
    rho = R/R(1);

    % integral time scale = area under rho up to its first zero crossing
    % this is roughly how long the turbulence stays correlated with itself
    iz = find(rho <= 0, 1);
    if isempty(iz)
        iz = L+1;
    end
    Tint(k) = sum(rho(1:iz-1))/fs;

    % store the legend label for this averaging length
    leg{k} = sprintf('M = %d', M);

    % R(0) = bar(u'u'), so the zero-lag value is the variance of this segment
    fprintf('%7d       %9.4f            %9.2f\n', M, R(1), Tint(k))

    % plot the requested autocorrelation
    % zero lag is skipped because a logarithmic x-axis cannot display log(0)
    semilogx(lag(2:end), R(2:end), 'LineWidth', 1.2)
    hold on
end

% label the autocorrelation plot
xlabel('lag \Delta t [s]')
ylabel('$\overline{u''(t)u''(t+\Delta t)}$ [m$^2$/s$^2$]', ...
    'Interpreter', 'latex')
legend(leg, 'Location', 'best')
title('Autocorrelation for different averaging lengths')
grid on



%% ITEM g

% use 50 histogram bins to estimate the pdf of u over the whole record
nb = 50;

% count how many velocity measurements fall inside each bin
[cnt, edges] = histcounts(u, nb);

% calculate the width of each bin
du = edges(2) - edges(1);

% calculate the center velocity of each bin
uc = edges(1:end-1) + du/2;

% convert the histogram counts into a probability density
% dividing by N gives probability and dividing by du gives density
pdfu = cnt/(N*du);

% check property 1: integral of pdf(u) du = 1
% the continuous integral is approximated by a sum over the bins
area = sum(pdfu*du);

% check property 2: bar(u) = integral of u*pdf(u) du
% each bin is represented by its center velocity
mean_pdf = sum(uc.*pdfu*du);

fprintf('\n---- ITEM g ----\n')
fprintf('integral pdf(u) du = %.6f\n', area)
fprintf('mean from data     = %.6f m/s\n', ubar)
fprintf('integral u pdf du  = %.6f m/s\n', mean_pdf)
fprintf('difference in means = %.3e m/s\n', mean_pdf - ubar)

% plot the estimated probability density function
figure
bar(uc, pdfu, 1)
xlabel('u [m/s]')
ylabel('pdf(u) [s/m]')
title(sprintf('PDF of u, %d bins', nb))
grid on



%% OPTIONAL: PDF check for different numbers of bins

% repeat the PDF checks using several different histogram resolutions
% this shows how representing all measurements in a bin by the bin center
% affects the PDF-based estimate of the mean
fprintf('\n bins     du      integral    mean from pdf   error in mean\n')

for nb2 = [5 10 25 50 100 500]

    % calculate a histogram using nb2 bins
    [c2, e2] = histcounts(u, nb2);

    % calculate the bin width
    d2 = e2(2) - e2(1);

    % calculate the center of each bin
    centers2 = e2(1:end-1) + d2/2;

    % convert histogram counts to a probability density
    p2 = c2/(N*d2);

    % calculate the mean from the discretized PDF
    m2 = sum(centers2 .* p2 * d2);

    % print the normalization and PDF-based mean for comparison
    fprintf('%5d  %7.4f   %9.6f   %10.5f    %10.2e\n', ...
        nb2, d2, sum(p2*d2), m2, m2 - ubar)
end



%% OPTIONAL: Gaussian comparison

% calculate the population standard deviation of the measured velocity
sig = std(u,1);

% create a smooth range of velocities for plotting a Gaussian curve
ug = linspace(min(u), max(u), 300);

% calculate a Gaussian PDF with the same mean and standard deviation
% this is only a visual comparison; the HW does not assume u is Gaussian
pg = exp(-(ug - ubar).^2/(2*sig^2))/(sig*sqrt(2*pi));

% compare the measured PDF with the Gaussian distribution
figure
bar(uc, pdfu, 1); hold on
plot(ug, pg, 'r', 'LineWidth', 1.5)
xlabel('u [m/s]')
ylabel('pdf(u) [s/m]')
legend('pdf from data', 'Gaussian, same mean & std', ...
    'Location', 'best')
title('Measured PDF compared with Gaussian distribution')
grid on



%% running mean with periodic wrap

function xm = runmean_periodic(x, n)

    % make sure the input data are a column vector
    x = x(:);

    % number of points in the full data record
    N = length(x);

    % determine how many points are used on each side of the current location
    % for an even window, one side has one extra point
    nL = floor((n-1)/2);
    nR = n - 1 - nL;

    % create periodic indices so values past either end wrap to the other side
    % this implements the periodic boundary condition allowed in items c and d
    idx = mod(-nL:N-1+nR, N) + 1;

    % calculate a cumulative sum of the periodically extended data
    % this allows the running average to be calculated efficiently
    c = cumsum([0; x(idx)]);

    % difference between cumulative sums gives each n-point window sum
    % dividing by n gives the running mean
    xm = (c(n+1:end) - c(1:end-n))/n;

    % return the result as a column vector
    xm = xm(:);

end