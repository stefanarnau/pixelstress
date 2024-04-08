clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';
PATH_TF_DATA = '/mnt/data_dump/pixelstress/3_tf_data/';
PATH_FIELDTRIP   = '/home/plkn/fieldtrip-master/';

% List of preprocessed datasets
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
                '40_1',...
                '41_2',...
                '42_2',...
               };

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Init ft
addpath(PATH_FIELDTRIP);
ft_defaults;

% Load stuff
load([PATH_TF_DATA, 'group_idx.mat']);
load([PATH_TF_DATA, 'tf_times.mat']);
load([PATH_TF_DATA, 'tf_freqs.mat']);
load([PATH_TF_DATA, 'chanlocs.mat']);

% Build elec struct from chanlocs
for ch = 1 : numel(chanlocs)
    elec.label{ch} = chanlocs(ch).labels;
    elec.elecpos(ch, :) = [chanlocs(ch).X, chanlocs(ch).Y, chanlocs(ch).Z];
    elec.chanpos(ch, :) = [chanlocs(ch).X, chanlocs(ch).Y, chanlocs(ch).Z];
end

% Prepare layout
cfg      = [];
cfg.elec = elec;
cfg.rotate = 90;
layout = ft_prepare_layout(cfg);


% Loop subjects and load ersps
for s = 1 : length(subject_list)

    load([PATH_TF_DATA, 'vp_', subject_list{s}(1 : end - 2),'_ersps.mat']);

    ersp_close.powspctrm = squeeze(ersps(1, :, :, :));
    ersp_close.dimord    = 'chan_freq_time';
    ersp_close.label     = elec.label;
    ersp_close.freq      = tf_freqs;
    ersp_close.time      = tf_times;

    ersp_below.powspctrm = squeeze(ersps(2, :, :, :));
    ersp_below.dimord    = 'chan_freq_time';
    ersp_below.label     = elec.label;
    ersp_below.freq      = tf_freqs;
    ersp_below.time      = tf_times;

    ersp_above.powspctrm = squeeze(ersps(3, :, :, :));
    ersp_above.dimord    = 'chan_freq_time';
    ersp_above.label     = elec.label;
    ersp_above.freq      = tf_freqs;
    ersp_above.time      = tf_times;

    tf_data_close{s} = ersp_close;
    tf_data_below{s} = ersp_below;
    tf_data_above{s} = ersp_above;

end

% Calculate grand averages
cfg = [];
cfg.keepindividual = 'yes';
GA_close = ft_freqgrandaverage(cfg, tf_data_close{1, :});
GA_below = ft_freqgrandaverage(cfg, tf_data_below{1, :});
GA_above = ft_freqgrandaverage(cfg, tf_data_above{1, :});

% Define neighbours
cfg                 = [];
cfg.layout          = layout;
cfg.feedback        = 'no';
cfg.method          = 'triangulation'; 
cfg.neighbours      = ft_prepare_neighbours(cfg, GA_close);
neighbours          = cfg.neighbours;


idx_chan = [9, 10, 65];
pd_close = squeeze(mean(GA_close.powspctrm(:, idx_chan, :, :), [1, 2]));
pd_below = squeeze(mean(GA_below.powspctrm(:, idx_chan, :, :), [1, 2]));
pd_above = squeeze(mean(GA_above.powspctrm(:, idx_chan, :, :), [1, 2]));

figure()
clims = [-5, 5];
cmap = 'jet';

subplot(1, 3, 1)
pd = pd_close;
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'log', 'YTick', [4, 8, 12, 20])
colorbar;
title('close', 'FontSize', 10)

subplot(1, 3, 2)
pd = pd_below;
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'log', 'YTick', [4, 8, 12, 20])
colorbar;
title('below', 'FontSize', 10)

subplot(1, 3, 3)
pd = pd_above;
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'log', 'YTick', [4, 8, 12, 20])
colorbar;
title('above', 'FontSize', 10)



% Testparams
testalpha   = 0.025;
voxelalpha  = 0.01;



% Set config. Same for all tests
cfg = [];
cfg.tail             = 1;
cfg.statistic        = 'depsamplesFmultivariate';
cfg.alpha            = testalpha;
cfg.neighbours       = neighbours;
cfg.minnbchan        = 2;
cfg.method           = 'montecarlo';
cfg.correctm         = 'cluster';
cfg.clustertail      = 1;
cfg.clusteralpha     = voxelalpha;
cfg.clusterstatistic = 'maxsum';
cfg.numrandomization = 500;
cfg.computecritval   = 'yes'; 
cfg.ivar             = 1;
cfg.uvar             = 2;

% Set up design
n_subjects = length(subject_list);
design = zeros(2, n_subjects * 3);
design(1, :) = [ones(1, n_subjects), ones(1, n_subjects) * 2, ones(1, n_subjects) * 3];
design(2, :) = [1 : n_subjects, 1 : n_subjects, 1 : n_subjects];
cfg.design = design;

% The test
[stat_traject]  = ft_freqstatistics(cfg, GA_close, GA_below, GA_above);




