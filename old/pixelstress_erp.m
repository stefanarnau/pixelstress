clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';
PATH_OUT = '/mnt/data_dump/pixelstress/3_behavior/';

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
               };

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Load ERP info
EEG_ERP_INFO = pop_loadset('filename', ['vp_', subject_list{1}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Init erp matrix
erps = zeros(length(subject_list), 4, EEG_ERP_INFO.nbchan, EEG_ERP_INFO.pnts);

% Table for between factor
group_idx = [];

% Loop subjects
for s = 1 : length(subject_list)

    % Load subject ERP data
    EEG = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Get between subject condition (1=exp, 2=control)
    group_idx(s) = EEG.trialinfo.session_condition(1);

    % Get trial idx
    idx_close_hi_earl = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr <= 6;
    idx_close_lo_earl = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr <= 6;
    idx_clear_hi_earl = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr <= 6;
    idx_clear_lo_earl = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr <= 6;
    idx_close_hi_late = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr >= 7;
    idx_close_lo_late = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr >= 7;
    idx_clear_hi_late = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr >= 7;
    idx_clear_lo_late = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr >= 7;

    % Get ERP
    erps(s, 1, :, :) = squeeze(mean(EEG.data(:, :, idx_close_hi_late), 3));
    erps(s, 2, :, :) = squeeze(mean(EEG.data(:, :, idx_close_lo_late), 3));
    erps(s, 3, :, :) = squeeze(mean(EEG.data(:, :, idx_clear_hi_late), 3));
    erps(s, 4, :, :) = squeeze(mean(EEG.data(:, :, idx_clear_lo_late), 3));

    erp_times = EEG.times;

end

% Average erp for groups
erps_exp =  squeeze(mean(erps(group_idx == 1, :, :, :), 1));
erps_cnt =  squeeze(mean(erps(group_idx == 2, :, :, :), 1));

% Set topo times
idx_topo_times = erp_times >= -500 & erp_times <= 0;
topo_clim = [-3, 3];

% Plot topos exp
figure()
subplot(2, 2, 1)
pd = squeeze(mean(squeeze(erps_exp(1, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('exp close hi')
subplot(2, 2, 2)
pd = squeeze(mean(squeeze(erps_exp(2, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('exp close lo')
subplot(2, 2, 3)
pd = squeeze(mean(squeeze(erps_exp(3, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('exp clear hi')
subplot(2, 2, 4)
pd = squeeze(mean(squeeze(erps_exp(4, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('exp clear lo')

% Plot topos cnt
figure()
subplot(2, 2, 1)
pd = squeeze(mean(squeeze(erps_cnt(1, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('cnt close hi')
subplot(2, 2, 2)
pd = squeeze(mean(squeeze(erps_cnt(2, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('cnt close lo')
subplot(2, 2, 3)
pd = squeeze(mean(squeeze(erps_cnt(3, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('cnt clear hi')
subplot(2, 2, 4)
pd = squeeze(mean(squeeze(erps_cnt(4, :, idx_topo_times)), 2));
topoplot(pd, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', topo_clim)
title('cnt clear lo')

% Average frontal erp
idx_frontal = [9, 10, 65];
idx_frontal = [48, 49, 14];
idx_frontal = [20, 53, 19, 48, 49, 14, 9, 10, 65];
erp_frontal_exp = squeeze(mean(erps_exp(:, idx_frontal, :), 2));
erp_frontal_cnt = squeeze(mean(erps_cnt(:, idx_frontal, :), 2));

% Plot
figure()
subplot(1, 2, 1)
plot(erp_times, erp_frontal_exp(1, :), 'k-', 'LineWidth', 1.5)
hold on
plot(erp_times, erp_frontal_exp(2, :), 'b-', 'LineWidth', 1.5)
plot(erp_times, erp_frontal_exp(3, :), 'r-', 'LineWidth', 1.5)
plot(erp_times, erp_frontal_exp(4, :), 'g-', 'LineWidth', 1.5)
ylim([-4, 2])
grid on
title('EXP')

subplot(1, 2, 2)
plot(erp_times, erp_frontal_cnt(1, :), 'k:', 'LineWidth', 1.5)
hold on
plot(erp_times, erp_frontal_cnt(2, :), 'b:', 'LineWidth', 1.5)
plot(erp_times, erp_frontal_cnt(3, :), 'r:', 'LineWidth', 1.5)
plot(erp_times, erp_frontal_cnt(4, :), 'g:', 'LineWidth', 1.5)
ylim([-4, 2])
grid on
legend({'close hi', 'close lo', 'clear hi', 'clear lo'})
title('CNT')

% Create result table
cnv_table = [];
idx_time = erp_times >= -500 & erp_times <= 0;
tmp = squeeze(mean(erps(:, :, idx_frontal, idx_time), [3, 4]));

% Loop subjects
counter = 0;
for s = 1 : length(subject_list)

    % Loop within conditions
    for wthcnd = 1 : 4

        % Fill
        counter = counter + 1;
        cnv_table(counter, :) = [str2double(subject_list{s}(1 : end - 2)), group_idx(s), wthcnd <= 2, mod(wthcnd, 2), tmp(s, wthcnd)];

    end

end

cnv_table = array2table(cnv_table, 'VariableNames', {'id', 'group', 'dist', 'outcome', 'cnv_amp'});
writetable(cnv_table, [PATH_OUT, 'cnv_table.csv']);