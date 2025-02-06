clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2024.0/';
PATH_FIELDTRIP = '/home/plkn/fieldtrip-master/';
PATH_OUTPUT = '/mnt/data_dump/pixelstress/plots/';

% The list
subject_list = {'9_1',...
                '10_1',...
                '14_1',...
                '17_1',...
                '19_1',...
                '21_1',...
                '23_1',...
                '25_1',...
                '26_1',...
                '28_1',...
                '30_1',...
                '32_1',...
                '35_1',...
                '36_1',...
                '38_1',...
                '40_1',...
                '43_1',...
                '44_1',...
                '47_1',...
                '48_1',...
                '50_1',...
                '52_1',...
                '55_1',...
                '57_1',...
                '58_1',...
                '76_1',...
                '77_1',...
                '79_1',...
                '81_1',...
                '82_1',...
                '84_1',...
                '85_1',...
                '86_1',...
               };

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Init ft
addpath(PATH_FIELDTRIP);
ft_defaults;

% Load ERP info
EEG_ERP_INFO = pop_loadset('filename', ['vp_', subject_list{1}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Crop times
time_idx = EEG_ERP_INFO.times >= -1700 & EEG_ERP_INFO.times <= 50;

% Init some nice matrices
erps_close = zeros(length(subject_list), sum(time_idx), EEG_ERP_INFO.nbchan);
erps_below = zeros(length(subject_list), sum(time_idx), EEG_ERP_INFO.nbchan);
erps_above = zeros(length(subject_list), sum(time_idx), EEG_ERP_INFO.nbchan);
coefs_total = zeros(length(subject_list), sum(time_idx), EEG_ERP_INFO.nbchan);
coefs_close = zeros(length(subject_list), sum(time_idx), EEG_ERP_INFO.nbchan);
coefs_below = zeros(length(subject_list), sum(time_idx), EEG_ERP_INFO.nbchan);
coefs_above = zeros(length(subject_list), sum(time_idx), EEG_ERP_INFO.nbchan);
coefs_fake = zeros(length(subject_list), sum(time_idx), EEG_ERP_INFO.nbchan);

% Loop subjects and calculate condition ERPs
for s = 1 : length(subject_list)

    % Load subject ERP data
    EEG = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % New electrode order
    new_order = [1, 2,...          % FP1 Fp2
                 33, 34, 35, 36,...          % AF7 AF3 AF4 AF8 
                 3, 37, 4, 38, 5, 39, 6, 40, 7,...   % F7 F5 F3 F1 Fz F2 F4 F6 F8
                 41, 42, 8, 43, 9, 65, 10, 44, 11, 45, 46,...           % FT9 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 FT10 
                 12, 47, 13, 48, 14, 49, 15, 50, 16,...   % T7 C5 C3 C1 Cz C2 C4 C6 T8
                 17, 51, 18, 52, 19, 53, 20, 54, 21, 55, 22,...           % TP9 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 TP10
                 23, 56, 24, 57, 25, 58, 26, 59, 27,... % P7 P5 P3 P1 Pz P2 P4 P6 P8
                 28, 60, 61, 62, 63, 64, 32,...         % PO9 PO7 PO3 POz PO4 PO8 PO10
                 29, 30, 31];           % O1 Oz O2

    % Re-order channels
    eeg_data = EEG.data(new_order, :, :);
    chanlocs = EEG.chanlocs;
    for ch = 1 : numel(EEG.chanlocs)
        chanlocs(ch) = EEG.chanlocs(new_order(ch));
    end

    % Apply moving average
    %eeg_data = movmean(eeg_data, 100, 2);

    % Crop in time
    eeg_times = EEG.times(time_idx);
    eeg_data = eeg_data(:, time_idx, :);

    % Get trial idx
    idx_close = EEG.trialinfo.block_wiggleroom == 0;
    idx_below = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1;
    idx_above = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1;

    % Add column to trialinfo: trial nr in block
    EEG.trialinfo.trial_nr_in_block = ((EEG.trialinfo.sequence_nr - 1) * 8) + EEG.trialinfo.trial_nr;

    % Regression design matrices
    desmat_total = [ones(EEG.trials, 1), EEG.trialinfo.trial_nr_in_block];
    desmat_close = [ones(sum(idx_close), 1), EEG.trialinfo.trial_nr_in_block(idx_close)];
    desmat_below = [ones(sum(idx_below), 1), EEG.trialinfo.trial_nr_in_block(idx_below)];
    desmat_above = [ones(sum(idx_above), 1), EEG.trialinfo.trial_nr_in_block(idx_above)];

    % Scale predictors
    desmat_total(:, 2) = zscore(desmat_total(:, 2));
    desmat_close(:, 2) = zscore(desmat_close(:, 2));
    desmat_below(:, 2) = zscore(desmat_below(:, 2));
    desmat_above(:, 2) = zscore(desmat_above(:, 2));

    % Permute data to trials x times x channels
    eeg_data_total = permute(eeg_data, [3, 2, 1]);

    % Subset
    eeg_data_close = eeg_data_total(idx_close, :, :);
    eeg_data_below = eeg_data_total(idx_below, :, :);
    eeg_data_above = eeg_data_total(idx_above, :, :);

    % reshape
    d_total = reshape(eeg_data_total, size(eeg_data_total, 1), size(eeg_data_total, 2) * size(eeg_data_total, 3));
    d_close = reshape(eeg_data_close, size(eeg_data_close, 1), size(eeg_data_close, 2) * size(eeg_data_close, 3));
    d_below = reshape(eeg_data_below, size(eeg_data_below, 1), size(eeg_data_below, 2) * size(eeg_data_below, 3));
    d_above = reshape(eeg_data_above, size(eeg_data_above, 1), size(eeg_data_above, 2) * size(eeg_data_above, 3));

    % Normalize
    d_total = zscore(d_total, [], 1);
    d_close = zscore(d_close, [], 1);
    d_below = zscore(d_below, [], 1);
    d_above = zscore(d_above, [], 1);

    % OLS fit
    tmp = (desmat_total' * desmat_total) \ desmat_total' * d_total;
    coefs_total(s, :, :) = reshape(squeeze(tmp(2, :)), [sum(time_idx), EEG.nbchan]);

    tmp = (desmat_close' * desmat_close) \ desmat_close' * d_close;
    coefs_close(s, :, :) = reshape(squeeze(tmp(2, :)), [sum(time_idx), EEG.nbchan]);

    tmp = (desmat_below' * desmat_below) \ desmat_below' * d_below;
    coefs_below(s, :, :) = reshape(squeeze(tmp(2, :)), [sum(time_idx), EEG.nbchan]);

    tmp = (desmat_above' * desmat_above) \ desmat_above' * d_above;
    coefs_above(s, :, :) = reshape(squeeze(tmp(2, :)), [sum(time_idx), EEG.nbchan]);

    % Generate null hypothesis coefs for main effect
    fakedesmat = desmat_total;
    fakedesmat(:, 2) = desmat_total(randperm(size(desmat_total, 1)), 2);
    tmp = (fakedesmat' * fakedesmat) \ fakedesmat' * d_total;
    coefs_fake(s, :, :) = reshape(squeeze(tmp(2, :)), [sum(time_idx), EEG.nbchan]);

    % Save trajectory erps
    erps_close(s, :, :) = squeeze(mean(eeg_data_close, 1));
    erps_below(s, :, :) = squeeze(mean(eeg_data_below, 1));
    erps_above(s, :, :) = squeeze(mean(eeg_data_above, 1));

end

% Plot all chans
figure()
subplot(3, 1, 1)
contourf(eeg_times, [1:65], squeeze(mean(erps_close, 1))', 40, 'linecolor','none');
clim([-3, 3])
colorbar()
subplot(3, 1, 2)
contourf(eeg_times, [1:65], squeeze(mean(erps_below, 1))', 40, 'linecolor','none');
clim([-3, 3])
colorbar()
subplot(3, 1, 3)
contourf(eeg_times, [1:65], squeeze(mean(erps_above, 1))', 40, 'linecolor','none');
clim([-3, 3])
colorbar()
colormap('jet')

% Plot midline ERPs
figure()
subplot(3, 1, 1)
chan_idx = 11;
plot(eeg_times, mean(squeeze(erps_close(:, :, chan_idx)), 1), 'LineWidth', 1.5)
hold on
plot(eeg_times, mean(squeeze(erps_below(:, :, chan_idx)), 1), 'LineWidth', 1.5)
plot(eeg_times, mean(squeeze(erps_above(:, :, chan_idx)), 1), 'LineWidth', 1.5)
hold off
legend({'close', 'below', 'above'})
title(chanlocs(chan_idx).labels)
subplot(3, 1, 2)
chan_idx = 21;
plot(eeg_times, mean(squeeze(erps_close(:, :, chan_idx)), 1), 'LineWidth', 1.5)
hold on
plot(eeg_times, mean(squeeze(erps_below(:, :, chan_idx)), 1), 'LineWidth', 1.5)
plot(eeg_times, mean(squeeze(erps_above(:, :, chan_idx)), 1), 'LineWidth', 1.5)
hold off
legend({'close', 'below', 'above'})
title(chanlocs(chan_idx).labels)
subplot(3, 1, 3)
chan_idx = 31;
plot(eeg_times, mean(squeeze(erps_close(:, :, chan_idx)), 1), 'LineWidth', 1.5)
hold on
plot(eeg_times, mean(squeeze(erps_below(:, :, chan_idx)), 1), 'LineWidth', 1.5)
plot(eeg_times, mean(squeeze(erps_above(:, :, chan_idx)), 1), 'LineWidth', 1.5)
hold off
legend({'close', 'below', 'above'})
title(chanlocs(chan_idx).labels)

% Plot regression coefs
figure()
subplot(3, 1, 1)
contourf(eeg_times, [1:65], squeeze(mean(coefs_close, 1))', 40, 'linecolor','none');
clim([-0.1, 0.1])
colorbar()
subplot(3, 1, 2)
contourf(eeg_times, [1:65], squeeze(mean(coefs_below, 1))', 40, 'linecolor','none');
clim([-0.1, 0.1])
colorbar()
subplot(3, 1, 3)
contourf(eeg_times, [1:65], squeeze(mean(coefs_above, 1))', 40, 'linecolor','none');
clim([-0.1, 0.1])
colorbar()
colormap('jet')

% Build elec struct
elec = struct();
for ch = 1 : length(EEG.chanlocs)
    elec.label{ch} = EEG.chanlocs(ch).labels;
    elec.elecpos(ch, :) = [EEG.chanlocs(ch).X, EEG.chanlocs(ch).Y, EEG.chanlocs(ch).Z];
    elec.chanpos(ch, :) = [EEG.chanlocs(ch).X, EEG.chanlocs(ch).Y, EEG.chanlocs(ch).Z];
end

% Build GAs for main effect trajectories
for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = eeg_times;
    d.avg = squeeze(erps_close(s, :, :))';
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_close = ft_timelockgrandaverage(cfg, D{1, :});

for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = eeg_times;
    d.avg = squeeze(erps_below(s, :, :))';
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_below = ft_timelockgrandaverage(cfg, D{1, :});

for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = eeg_times;
    d.avg = squeeze(erps_above(s, :, :))';
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_above = ft_timelockgrandaverage(cfg, D{1, :});

% Build GAs for main effect stage
for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = eeg_times;
    d.avg = squeeze(coefs_total(s, :, :))';
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_coefs_total = ft_timelockgrandaverage(cfg, D{1, :});

for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = eeg_times;
    d.avg = squeeze(coefs_fake(s, :, :))';
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_coefs_fake = ft_timelockgrandaverage(cfg, D{1, :});


% Build GAs for interactions
for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = eeg_times;
    d.avg = squeeze(coefs_close(s, :, :))';
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_coefs_close = ft_timelockgrandaverage(cfg, D{1, :});

for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = eeg_times;
    d.avg = squeeze(coefs_below(s, :, :))';
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_coefs_below = ft_timelockgrandaverage(cfg, D{1, :});

for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = eeg_times;
    d.avg = squeeze(coefs_above(s, :, :))';
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_coefs_above = ft_timelockgrandaverage(cfg, D{1, :});

% Prepare layout
cfg      = [];
cfg.elec = elec;
cfg.rotate = 90;
layout = ft_prepare_layout(cfg);

% Define neighbours
cfg                 = [];
cfg.layout          = layout;
cfg.feedback        = 'no';
cfg.method          = 'triangulation'; 
cfg.neighbours      = ft_prepare_neighbours(cfg, GA_above);
neighbours = cfg.neighbours;

% Testparams
testalpha  = 0.025;
voxelalpha  = 0.05;
nperm = 1000;

% Set config for within test of trajectory effect
cfg = [];
cfg.tail             = 1;
cfg.statistic        = 'depsamplesT';
cfg.alpha            = testalpha;
cfg.neighbours       = neighbours;
cfg.minnbchan        = 2;
cfg.method           = 'montecarlo';
cfg.correctm         = 'cluster';
cfg.clustertail      = 1;
cfg.clusteralpha     = voxelalpha;
cfg.clusterstatistic = 'maxsum';
cfg.numrandomization = nperm;
cfg.computecritval   = 'yes'; 
cfg.ivar             = 1;
cfg.uvar             = 2;
cfg.design           = [ones(1, numel(subject_list)), 2 * ones(1, numel(subject_list));...
                             1 : numel(subject_list), 1 : numel(subject_list)];

% The test for main effects trajectory
[stat_below_vs_above]  = ft_timelockstatistics(cfg, GA_below, GA_above);
[stat_below_vs_close]  = ft_timelockstatistics(cfg, GA_below, GA_close);
[stat_above_vs_close]  = ft_timelockstatistics(cfg, GA_above, GA_close);

maxval = max([max(abs(stat_below_vs_above.stat(:))),...
          max(abs(stat_below_vs_close.stat(:))),...
          max(abs(stat_above_vs_close.stat(:))),...
              ]);

% Plot time x space t-values
figure;
subplot(3, 1, 1)
pd = stat_below_vs_above.stat;
contourf(stat_below_vs_above.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_below_vs_above.time, [1 : 65], stat_below_vs_above.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('below vs above')
hold off;

subplot(3, 1, 2)
pd = stat_below_vs_close.stat;
contourf(stat_below_vs_close.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_below_vs_close.time, [1 : 65], stat_below_vs_close.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('below vs close')
hold off;

subplot(3, 1, 3)
pd = stat_above_vs_close.stat;
contourf(stat_above_vs_close.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_above_vs_close.time, [1 : 65], stat_above_vs_close.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('above vs close')
hold off;

% Testparams
testalpha  = 0.025;
voxelalpha  = 0.05;
nperm = 1000;

% Set config for within test of stage effect
cfg = [];
cfg.tail             = 1;
cfg.statistic        = 'depsamplesT';
cfg.alpha            = testalpha;
cfg.neighbours       = neighbours;
cfg.minnbchan        = 2;
cfg.method           = 'montecarlo';
cfg.correctm         = 'cluster';
cfg.clustertail      = 1;
cfg.clusteralpha     = voxelalpha;
cfg.clusterstatistic = 'maxsum';
cfg.numrandomization = nperm;
cfg.computecritval   = 'yes'; 
cfg.ivar             = 1;
cfg.uvar             = 2;
cfg.design           = [ones(1, numel(subject_list)), 2 * ones(1, numel(subject_list));...
                             1 : numel(subject_list), 1 : numel(subject_list)];

% The test for main effect stage
[stat_stage]  = ft_timelockstatistics(cfg, GA_coefs_total, GA_coefs_fake);

maxval = max(abs(stat_stage.stat(:)));

% Plot time x space t-values
figure;
pd = stat_stage.stat;
contourf(stat_stage.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_stage.time, [1 : 65], stat_stage.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('stage')
hold off;


% Testparams
testalpha  = 0.025;
voxelalpha  = 0.05;
nperm = 1000;

% Set config for within test of trajectory effect
cfg = [];
cfg.tail             = 1;
cfg.statistic        = 'depsamplesT';
cfg.alpha            = testalpha;
cfg.neighbours       = neighbours;
cfg.minnbchan        = 2;
cfg.method           = 'montecarlo';
cfg.correctm         = 'cluster';
cfg.clustertail      = 1;
cfg.clusteralpha     = voxelalpha;
cfg.clusterstatistic = 'maxsum';
cfg.numrandomization = nperm;
cfg.computecritval   = 'yes'; 
cfg.ivar             = 1;
cfg.uvar             = 2;
cfg.design           = [ones(1, numel(subject_list)), 2 * ones(1, numel(subject_list));...
                             1 : numel(subject_list), 1 : numel(subject_list)];

% The test for interaction
[stat_int_below_vs_above]  = ft_timelockstatistics(cfg, GA_coefs_below, GA_coefs_above);
[stat_int_below_vs_close]  = ft_timelockstatistics(cfg, GA_coefs_below, GA_coefs_close);
[stat_int_above_vs_close]  = ft_timelockstatistics(cfg, GA_coefs_above, GA_coefs_close);

maxval = max([max(abs(stat_int_below_vs_above.stat(:))),...
              max(abs(stat_int_below_vs_close.stat(:))),...
              max(abs(stat_int_above_vs_close.stat(:))),...
              ]);

% Plot time x space t-values
figure;
subplot(3, 1, 1)
pd = stat_int_below_vs_above.stat;
contourf(stat_int_below_vs_above.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_int_below_vs_above.time, [1 : 65], stat_int_below_vs_above.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('int below vs above')
hold off;

subplot(3, 1, 2)
pd = stat_int_below_vs_close.stat;
contourf(stat_int_below_vs_close.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_int_below_vs_close.time, [1 : 65], stat_int_below_vs_close.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('int below vs close')
hold off;

subplot(3, 1, 3)
pd = stat_int_above_vs_close.stat;
contourf(stat_int_above_vs_close.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_int_above_vs_close.time, [1 : 65], stat_int_above_vs_close.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('int above vs close')
hold off;
aa = bb;








% The test for main effect stage
[stat_stage]  = ft_timelockstatistics(cfg, GA_early, GA_late);

maxval = max(abs(stat_trajectory_below_vs_above.stat(:)));

% Plot time x space t-values
figure;
pd = stat_stage.stat;
contourf(stat_stage.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_stage.time, [1 : 65], stat_stage.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('early vs late')
hold off;


% The test for main effects trajectory
[stat_interaction_below_vs_above]  = ft_timelockstatistics(cfg, GA_int_below, GA_int_above);
[stat_interaction_below_vs_close]  = ft_timelockstatistics(cfg, GA_int_below, GA_int_close);
[stat_interaction_above_vs_close]  = ft_timelockstatistics(cfg, GA_int_above, GA_int_close);

maxval = max([max(abs(stat_interaction_below_vs_above.stat(:))),...
              max(abs(stat_interaction_below_vs_close.stat(:))),...
              max(abs(stat_interaction_above_vs_close.stat(:))),...
              ]);

% Plot time x space t-values
figure;
subplot(3, 1, 1)
pd = stat_interaction_below_vs_above.stat;
contourf(stat_interaction_below_vs_above.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_interaction_below_vs_above.time, [1 : 65], stat_interaction_below_vs_above.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('int below vs above')
hold off;

subplot(3, 1, 2)
pd = stat_interaction_below_vs_close.stat;
contourf(stat_interaction_below_vs_close.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_interaction_below_vs_close.time, [1 : 65], stat_interaction_below_vs_close.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('int below vs close')
hold off;

subplot(3, 1, 3)
pd = stat_interaction_above_vs_close.stat;
contourf(stat_interaction_above_vs_close.time, [1 :65], pd, 40, 'linecolor','none')
hold on
contour(stat_interaction_above_vs_close.time, [1 : 65], stat_interaction_above_vs_close.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')
set(gca, 'clim', [-maxval, maxval])
colorbar;
title('int above vs close')
hold off;




aa=bb



% Set config for between test
cfg = [];
cfg.tail             = 0;
cfg.statistic        = 'indepsamplesT';
cfg.alpha            = testalpha;
cfg.neighbours       = neighbours;
cfg.minnbchan        = 2;
cfg.method           = 'montecarlo';
cfg.correctm         = 'cluster';
cfg.clustertail      = 0;
cfg.clusteralpha     = voxelalpha;
cfg.clusterstatistic = 'maxsum';
cfg.numrandomization = nperm;
cfg.computecritval   = 'yes'; 
cfg.ivar             = 1;
cfg.uvar             = 2;
cfg.design           = [ones(1, numel(subject_list)), 2 * ones(1, numel(subject_list)); 1 : numel(subject_list), 1 : numel(subject_list)];


% Set up design
cfg.design = group_idx;

% The tests
[stat_group]  = ft_timelockstatistics(cfg, GA_group1, GA_group2);
[stat_interaction1]  = ft_timelockstatistics(cfg, GA_group1_close_below, GA_group2_close_below);
[stat_interaction2]  = ft_timelockstatistics(cfg, GA_group1_below_above, GA_group2_below_above);
[stat_interaction3]  = ft_timelockstatistics(cfg, GA_group1_close_above, GA_group2_close_above);

% Save masks
dlmwrite([PATH_OUT, 'contour_trajectory.csv'], stat_traject.mask);
dlmwrite([PATH_OUT, 'contour_group.csv'], stat_group.mask);
dlmwrite([PATH_OUT, 'contour_interaction1.csv'], stat_interaction1.mask);
dlmwrite([PATH_OUT, 'contour_interaction2.csv'], stat_interaction2.mask);
dlmwrite([PATH_OUT, 'contour_interaction3.csv'], stat_interaction3.mask);

% Calculate effect sizes
n_chans = numel(EEG.chanlocs);
apes_trajectory = [];
apes_group = [];
apes_interaction1 = [];
apes_interaction2 = [];
apes_interaction3 = [];

for ch = 1 : n_chans

    df_effect = 2;
    petasq = (squeeze(stat_traject.stat(ch, :)) * df_effect) ./ ((squeeze(stat_traject.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_trajectory(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));

    df_effect = 1;
    petasq = (squeeze(stat_group.stat(ch, :)) * df_effect) ./ ((squeeze(stat_group.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_group(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 2));

    df_effect = 1;
    petasq = (squeeze(stat_interaction1.stat(ch, :)) * df_effect) ./ ((squeeze(stat_interaction1.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_interaction1(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 2));

    df_effect = 1;
    petasq = (squeeze(stat_interaction2.stat(ch, :)) * df_effect) ./ ((squeeze(stat_interaction2.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_interaction2(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 2));

    df_effect = 1;
    petasq = (squeeze(stat_interaction3.stat(ch, :)) * df_effect) ./ ((squeeze(stat_interaction3.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_interaction3(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 2));

end

% Save effect sizes
dlmwrite([PATH_OUT, 'apes_trajectory.csv'], apes_trajectory);
dlmwrite([PATH_OUT, 'apes_trajectory.csv'], apes_group);
dlmwrite([PATH_OUT, 'apes_interaction1.csv'], apes_interaction1);
dlmwrite([PATH_OUT, 'apes_interaction2.csv'], apes_interaction2);
dlmwrite([PATH_OUT, 'apes_interaction3.csv'], apes_interaction3);

% Plot masks
figure()
subplot(2, 3, 1)
contourf(erp_times,[1:65], apes_trajectory, 50, 'LineColor', 'none')
caxis([-0.5, 0.5])
colormap(jet)
hold on
contour(erp_times,[1:65], stat_traject.mask, 'levels', 1, 'LineColor', 'k', 'LineWidth', 1.5)
title('trajectory')

subplot(2, 3, 2)
contourf(erp_times,[1:65], apes_group, 50, 'LineColor', 'none')
caxis([-0.5, 0.5])
colormap(jet)
hold on
contour(erp_times,[1:65], stat_group.mask, 'levels', 1, 'LineColor', 'k', 'LineWidth', 1.5)
title('group')

subplot(2, 3, 3)
contourf(erp_times,[1:65], apes_interaction1, 50, 'LineColor', 'none')
caxis([-0.5, 0.5])
colormap(jet)
hold on
contour(erp_times,[1:65], stat_interaction1.mask, 'levels', 1, 'LineColor', 'k', 'LineWidth', 1.5)
title('interaction1')

subplot(2, 3, 4)
contourf(erp_times,[1:65], apes_interaction2, 50, 'LineColor', 'none')
caxis([-0.5, 0.5])
colormap(jet)
hold on
contour(erp_times,[1:65], stat_interaction2.mask, 'levels', 1, 'LineColor', 'k', 'LineWidth', 1.5)
title('interaction2')

subplot(2, 3, 5)
contourf(erp_times,[1:65], apes_interaction3, 50, 'LineColor', 'none')
caxis([-0.5, 0.5])
colormap(jet)
hold on
contour(erp_times,[1:65], stat_interaction3.mask, 'levels', 1, 'LineColor', 'k', 'LineWidth', 1.5)
title('interaction3')

aa = bb













% Average erp for groups
erps_exp =  squeeze(mean(erps(group_idx == 1, :, :, :), 1));
erps_cnt =  squeeze(mean(erps(group_idx == 2, :, :, :), 1));

% Set topo times
idx_topo_times = erp_times >= -500 & erp_times <= 0;
topo_clim = [-3, 3];

% Plot topos exp
figure()
subplot(2, 3, 1)
pd = squeeze(mean(squeeze(erps_exp(1, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('exp close')
subplot(2, 3, 2)
pd = squeeze(mean(squeeze(erps_exp(2, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('exp below')
subplot(2, 3, 3)
pd = squeeze(mean(squeeze(erps_exp(3, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('exp above')
subplot(2, 3, 4)
pd = squeeze(mean(squeeze(erps_cnt(1, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('cnt close')
subplot(2, 3, 5)
pd = squeeze(mean(squeeze(erps_cnt(2, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('cnt below')
subplot(2, 3, 6)
pd = squeeze(mean(squeeze(erps_cnt(3, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('cnt above')



% Average frontal erp
idx_frontal = [48, 14, 49, 65, 53];
erp_frontal_exp = squeeze(mean(erps_exp(:, idx_frontal, :), 2));
erp_frontal_cnt = squeeze(mean(erps_cnt(:, idx_frontal, :), 2));

% Plot
figure()
subplot(1, 2, 1)
plot(erp_times, erp_frontal_exp(1, :), 'k-', 'LineWidth', 1.5)
hold on
plot(erp_times, erp_frontal_exp(2, :), 'b-', 'LineWidth', 1.5)
plot(erp_times, erp_frontal_exp(3, :), 'r-', 'LineWidth', 1.5)
ylim([-4, 2])
grid on
title('EXP')

subplot(1, 2, 2)
plot(erp_times, erp_frontal_cnt(1, :), 'k:', 'LineWidth', 1.5)
hold on
plot(erp_times, erp_frontal_cnt(2, :), 'b:', 'LineWidth', 1.5)
plot(erp_times, erp_frontal_cnt(3, :), 'r:', 'LineWidth', 1.5)
ylim([-4, 2])
grid on
legend({'close', 'below', 'above'})
title('CNT')

% Create result table
cnv_table = [];
idx_time = erp_times >= -500 & erp_times <= 0;
tmp = squeeze(mean(erps(:, :, idx_frontal, idx_time), [3, 4]));

% Loop subjects
counter = 0;
for s = 1 : length(subject_list)

    % Loop within conditions
    for wthcnd = 1 : 3

        % Fill
        counter = counter + 1;
        cnv_table(counter, :) = [str2double(subject_list{s}(1 : end - 2)), group_idx(s), wthcnd, tmp(s, wthcnd)];

    end

end

cnv_table = array2table(cnv_table, 'VariableNames', {'id', 'group', 'trajectory', 'cnv_amp'});
writetable(cnv_table, [PATH_OUT, 'cnv_table.csv']);