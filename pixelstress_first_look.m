clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';

% List of preprocessed datasets
subject_list = [2, 7, 8, 9, 10, 11 ,12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37];

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Load info
EEG = pop_loadset('filename', ['vp_', num2str(subject_list(1)), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Init erp matrix
erp = zeros(2, 8, EEG.pnts);

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
    idx_close_hi_earl = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == 1 & EEG.trialinfo.sequence_nr <= 6;
    idx_close_lo_earl = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr <= 6;
    idx_clear_hi_earl = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1 & EEG.trialinfo.sequence_nr <= 6;
    idx_clear_lo_earl = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr <= 6;
    idx_close_hi_late = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == 1 & EEG.trialinfo.sequence_nr >= 7;
    idx_close_lo_late = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr >= 7;
    idx_clear_hi_late = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1 & EEG.trialinfo.sequence_nr >= 7;
    idx_clear_lo_late = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr >= 7;

    % Channel idx
    chan_idx = [9, 10, 65];

    % Get erp
    erp(between_factor, 1, :) = squeeze(erp(between_factor, 1, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_close_hi_earl), [1, 3]))';
    erp(between_factor, 2, :) = squeeze(erp(between_factor, 2, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_close_lo_earl), [1, 3]))';
    erp(between_factor, 3, :) = squeeze(erp(between_factor, 3, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_clear_hi_earl), [1, 3]))';
    erp(between_factor, 4, :) = squeeze(erp(between_factor, 4, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_clear_lo_earl), [1, 3]))';
    erp(between_factor, 5, :) = squeeze(erp(between_factor, 5, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_close_hi_late), [1, 3]))';
    erp(between_factor, 6, :) = squeeze(erp(between_factor, 6, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_close_lo_late), [1, 3]))';
    erp(between_factor, 7, :) = squeeze(erp(between_factor, 7, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_clear_hi_late), [1, 3]))';
    erp(between_factor, 8, :) = squeeze(erp(between_factor, 8, :)) + squeeze(mean(EEG.data(chan_idx, :, idx_clear_lo_late), [1, 3]))';

end

% Scale
for f1 = 1 : 2
    for f2 = 1 : 8
        erp(f1, f2, :) = squeeze(erp(f1, f2, :)) ./ n_bcond(f1);
    end
end

% Plot
figure()
subplot(2, 2, 1)
plot(EEG.times, squeeze(erp(1, 1, :)), ':k', 'LineWidth', 1.5)
hold on
plot(EEG.times, squeeze(erp(1, 2, :)), ':r', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(1, 3, :)), 'k', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(1, 4, :)), 'r', 'LineWidth', 1.5)
title('experimental - early')

subplot(2, 2, 2)
plot(EEG.times, squeeze(erp(2, 1, :)), ':k', 'LineWidth', 1.5)
hold on
plot(EEG.times, squeeze(erp(2, 2, :)), ':r', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(2, 3, :)), 'k', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(2, 4, :)), 'r', 'LineWidth', 1.5)
title('erp control - early')

subplot(2, 2, 3)
plot(EEG.times, squeeze(erp(1, 5, :)), ':k', 'LineWidth', 1.5)
hold on
plot(EEG.times, squeeze(erp(1, 6, :)), ':r', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(1, 7, :)), 'k', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(1, 8, :)), 'r', 'LineWidth', 1.5)
title('erp experimental - late')

subplot(2, 2, 4)
plot(EEG.times, squeeze(erp(2, 5, :)), ':k', 'LineWidth', 1.5)
hold on
plot(EEG.times, squeeze(erp(2, 6, :)), ':r', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(2, 7, :)), 'k', 'LineWidth', 1.5)
plot(EEG.times, squeeze(erp(2, 8, :)), 'r', 'LineWidth', 1.5)
title('erp control - late')


legend({'close hi', 'close lo', 'clear hi', 'clear lo'})