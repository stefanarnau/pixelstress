clear all;

% PATH VARS
PATH_EEGLAB = '/home/plkn/eeglab2024.0/';
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

% A prepro-stats table
stats_abs = [];
stats_perc = [];

% Loop subjects
for s = 1 : length(subject_list)

    % Load erp data
    EEG_ERP = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

    % Load tf data
    EEG_TF = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

    % Fill stats
    stats_abs(s, :) = [EEG_ERP.trialinfo(1, 2).id, length(EEG_ERP.chans_rejected), length(EEG_ERP.rejected_epochs), length(EEG_ERP.nobrainer), length(EEG_TF.chans_rejected), length(EEG_TF.rejected_epochs), length(EEG_TF.nobrainer)];
    stats_perc(s, :) = [EEG_ERP.trialinfo(1, 2).id, length(EEG_ERP.chans_rejected) / 65, length(EEG_ERP.rejected_epochs) / 768, length(EEG_ERP.nobrainer) / 64, length(EEG_TF.chans_rejected) / 65, length(EEG_TF.rejected_epochs) / 768, length(EEG_TF.nobrainer) / 64];

end