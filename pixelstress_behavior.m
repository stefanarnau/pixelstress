clear all;

% PATH VARS
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';


subject_list = {'2_2',...
                '7_2',...
                '8_2',...
                '9_1',...
                '10_1',...
                '11_2',...
                '12_2',...
                '14_1',...
                '15_2',...
                '16_2',...
                '17_1',...
                '19_1',...
                '20_2',...
                '21_1',...
                '22_2',...
                '23_1',...
                '24_2',...
                '25_1',...
                '26_1',...
                '27_2',...
                '28_1',...
                '29_2',...
                '30_1',...
                '31_2',...
                '32_1',...
                '33_2',...
                '34_2',...
                '35_1',...
                '36_1',...
                '37_2',...
                '38_1',...
                '39_2',...
               };

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Collector matrix
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