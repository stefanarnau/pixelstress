clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';

% List of preprocessed datasets
subject_list = [2, 7, 8, 9, 10, 11 ,12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Load info
EEG = pop_loadset('filename', ['vp_', num2str(subject_list(1)), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Init erp matrix
erp = zeros(2, 4, EEG.pnts);

% Counter for subjects in between conditions
n_bcond = [0, 0];

% Loop subjects
for s = 1 : length(subject_list)

    % Load subject data. EEG data has dimensionality channels x times x trials
    EEG = pop_loadset('filename', ['vp_', num2str(subject_list(s)), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Get between subject condition (1=exp, 2=control)
    between_factor = EEG.trialinfo.session_condition(1);

    % Count subjects
    n_bcond(between_factor) = n_bcond(between_factor) + 1;

    % Get trial idx
    idx_close_hi = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == 1;
    idx_close_lo = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == -1;
    idx_clear_hi = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1;
    idx_clear_lo = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1;

    % Channel idx
    chan_idx = [65, 5, 9, 10];

    % Get erp
    erp(between_factor, 1, :) = squeeze(erp(between_factor, 1, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_close_hi), [1, 3]))';
    erp(between_factor, 2, :) = squeeze(erp(between_factor, 2, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_close_lo), [1, 3]))';
    erp(between_factor, 3, :) = squeeze(erp(between_factor, 3, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_clear_hi), [1, 3]))';
    erp(between_factor, 4, :) = squeeze(erp(between_factor, 4, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_clear_lo), [1, 3]))';

end

% Scale
for f1 = 1 : 2
    for f2 = 1 : 4
        erp(f1, f2, :) = squeeze(erp(f1, f2, :)) ./ n_bcond(f1);
    end
end

% Plot
figure()
subplot(1, 2, 1)
plot(EEG.times, squeeze(erp(1, 1, :)), ':k', 'LineWidth', 1.5)
hold on
plot(EEG.times, squeeze(erp(1, 2, :)), ':r', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(1, 3, :)), 'k', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(1, 4, :)), 'r', 'LineWidth', 1.5)
title('erp experimental')

subplot(1, 2, 2)
plot(EEG.times, squeeze(erp(2, 1, :)), ':k', 'LineWidth', 1.5)
hold on
plot(EEG.times, squeeze(erp(2, 2, :)), ':r', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(2, 3, :)), 'k', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(2, 4, :)), 'r', 'LineWidth', 1.5)
title('erp control')