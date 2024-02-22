clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';

% List of preprocessed datasets
subject_list = [2, 7, 8, 9, 10, 11 ,12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34];

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% A data matrix
data = [];

% Loop subjects and collect trialinfo
for s = 1 : length(subject_list)
    EEG = pop_loadset('filename', ['vp_', num2str(subject_list(s)), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');
    if s == 1
        data = EEG.trialinfo;
    else
        data = [data; EEG.trialinfo];
    end
end
