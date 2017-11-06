function demo_fgnsr
rng('default')
% Compare endmember extraction algorithms (EEAs) on the subsampled Urban
% hyperspecral image (every second pixel in each direction kept).
% 
% The EEAs under consideration are 
% VCA, SPA, XRAY, SNPA, H2NMF, and FGNSR. 
% 
% See also: Gillis & Luce, "A Fast Gradient Method for Nonnegative Sparse
% Regression with Self Dictionary"

close all

    addpath('hierclust2nmf_v2'); 
addpath('Endmember Extraction Algorithms');  

% 1. Load (undersampled) Urban data set into 'M'
load('urban_undersampled4.mat', 'M');

% Image size
dimim = 77;
% Number of pure pixels
r = 6;
% Number of subsampled pixels to be able to run NSRFG 
nbsubs = 100; 

% 2. Run experiment 
[Kall, Hall, resultsErr,Hall2, resultsErr2, nEEAs,resultsErr3,resultsErr4,resultsErr5] = experiment_EEAs( M , r  ) ; 

% 3. Display results 
fprintf('------------------------------------------------------ \n')
fprintf('  Algorithm   | Relative Error |   FC |   NNMP |   PWMP   |   AMP (%%) \n')
fprintf('------------------------------------------------------ \n')
for i = 1 : 5
    fprintf([nEEAs{i},'         |       %2.2f      |        %2.2f       |        %2.2f      |        %2.2f     |        %2.2f  \n'], resultsErr(i), resultsErr2(i), resultsErr3(i), resultsErr4(i), resultsErr5(i));
end


end
