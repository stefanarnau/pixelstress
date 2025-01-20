clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';
PATH_FIELDTRIP = '/home/plkn/fieldtrip-master/';

% List of preprocessed datasets
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
               };

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Init ft
addpath(PATH_FIELDTRIP);
ft_defaults;

% Load ERP info
EEG_ERP_INFO = pop_loadset('filename', ['vp_', subject_list{1}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Init erp matrix
erps = zeros(length(subject_list), 11, EEG_ERP_INFO.nbchan, EEG_ERP_INFO.pnts);

% Table for between factor
group_idx = [];

% Loop subjects and calculate condition ERPs
for s = 1 : length(subject_list)

    % Load subject ERP data
    EEG = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Get trial idx
    idx_close = EEG.trialinfo.sequence_nr >= 2 & EEG.trialinfo.block_wiggleroom == 0;
    idx_below = EEG.trialinfo.sequence_nr >= 2 & EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1;
    idx_above = EEG.trialinfo.sequence_nr >= 2 & EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1;
    
    idx_early = EEG.trialinfo.sequence_nr >= 2 & EEG.trialinfo.sequence_nr <= 5;
    idx_late  = EEG.trialinfo.sequence_nr >= 9 & EEG.trialinfo.sequence_nr <= 12;

    % Get ERP
    erps(s, 1, :, :) = squeeze(mean(EEG.data(:, :, idx_close), 3));
    erps(s, 2, :, :) = squeeze(mean(EEG.data(:, :, idx_below), 3));
    erps(s, 3, :, :) = squeeze(mean(EEG.data(:, :, idx_above), 3));

    erps(s, 4, :, :) = squeeze(mean(EEG.data(:, :, idx_early), 3));
    erps(s, 5, :, :) = squeeze(mean(EEG.data(:, :, idx_late), 3));

    erps(s, 6, :, :) = squeeze(mean(EEG.data(:, :, idx_close & idx_early), 3));
    erps(s, 7, :, :) = squeeze(mean(EEG.data(:, :, idx_close & idx_late), 3));
    erps(s, 8, :, :) = squeeze(mean(EEG.data(:, :, idx_below & idx_early), 3));
    erps(s, 9, :, :) = squeeze(mean(EEG.data(:, :, idx_below & idx_late), 3));
    erps(s, 10, :, :) = squeeze(mean(EEG.data(:, :, idx_above & idx_early), 3));
    erps(s, 11, :, :) = squeeze(mean(EEG.data(:, :, idx_above & idx_late), 3));

    erp_times = EEG.times;

end

% Crop in time
time_idx = EEG.times >= -1200 & EEG.times <= 500;
erp_times = EEG.times(time_idx);
erps = erps(:, :, :, time_idx);

% A plot


% Create y-axis data for 6 different lines
idx_chan = [5, 14, 65, 9, 10];
y1 = squeeze(mean(erps(:, 6,  idx_chan, :), [1, 3]));
y2 = squeeze(mean(erps(:, 7,  idx_chan, :), [1, 3]));
y3 = squeeze(mean(erps(:, 8,  idx_chan, :), [1, 3]));
y4 = squeeze(mean(erps(:, 9,  idx_chan, :), [1, 3]));
y5 = squeeze(mean(erps(:, 10, idx_chan, :), [1, 3]));
y6 = squeeze(mean(erps(:, 11, idx_chan, :), [1, 3]));

% Create the plot
figure;
plot(erp_times, y1, 'k-', 'LineWidth', 2);
hold on;
plot(erp_times, y2, 'k:', 'LineWidth', 2);
plot(erp_times, y3, 'r-', 'LineWidth', 2);
plot(erp_times, y4, 'r:', 'LineWidth', 2);
plot(erp_times, y5, 'g-', 'LineWidth', 2);
plot(erp_times, y6, 'g:', 'LineWidth', 2);

% Add labels and title
xlabel('X-axis');
ylabel('Y-axis');
title('Six Different Lines in One Plot');

% Add a legend
legend('close-e', 'close-l', 'below-e', 'below-l', 'above-e', 'below-l', 'Location', 'best');

% Add grid
grid on;

% Hold off to end the plotting
hold off;

aa = bb;




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
    d.time = erp_times;
    d.avg = squeeze(erps(s, 1, :, :));
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_close = ft_timelockgrandaverage(cfg, D{1, :});

for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = erp_times;
    d.avg = squeeze(erps(s, 2, :, :));
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_below = ft_timelockgrandaverage(cfg, D{1, :});

for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = erp_times;
    d.avg = squeeze(erps(s, 3, :, :));
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_above = ft_timelockgrandaverage(cfg, D{1, :});


% Build GAs for main effect trajectories
for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = erp_times;
    d.avg = squeeze(erps(s, 1, :, :));
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_close = ft_timelockgrandaverage(cfg, D{1, :});

for s = 1 : length(subject_list)
    d = [];
    d.dimord = 'chan_time';
    d.label = elec.label;
    d.time = erp_times;
    d.avg = squeeze(erps(s, 2, :, :));
    D{s} = d;
end
cfg=[];
cfg.keepindividual = 'yes';
GA_below = ft_timelockgrandaverage(cfg, D{1, :});





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
testalpha  = 0.05;
voxelalpha  = 0.05;
nperm = 1000;

% Set config for within test of trajectory effect
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
cfg.numrandomization = nperm;
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
[stat_traject]  = ft_timelockstatistics(cfg, GA_close, GA_below, GA_above);

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