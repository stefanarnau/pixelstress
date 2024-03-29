clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';

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
erp_frontal = zeros(length(subject_list), 8, EEG_ERP_INFO.pnts);

% Table for between factor
group_idx = [];

% Load TF info
EEG_TF_INFO = pop_loadset('filename', ['vp_', subject_list{1}(1 : end - 2), '_cleaned_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Set complex Morlet wavelet parameters
n_frq = 30;
frqrange = [2, 20];
tfres_range = [400, 100];

% Set wavelet time
wtime = -2 : 1 / EEG_TF_INFO.srate : 2;

% Determine fft frqs
hz = linspace(0, EEG_TF_INFO.srate, length(wtime));

% Create wavelet frequencies and tapering Gaussian widths in temporal domain
tf_freqs = linspace(frqrange(1), frqrange(2), n_frq);
fwhmTs = logspace(log10(tfres_range(1)), log10(tfres_range(2)), n_frq);

% Init matrices for wavelets
cmw = zeros(length(tf_freqs), length(wtime));
cmwX = zeros(length(tf_freqs), length(wtime));
tlim = zeros(1, length(tf_freqs));

% These will contain the wavelet widths as full width at 
% half maximum in the temporal and spectral domain
obs_fwhmT = zeros(1, length(tf_freqs));
obs_fwhmF = zeros(1, length(tf_freqs));

% Create the wavelets
for frq = 1 : length(tf_freqs)

    % Create wavelet with tapering gaussian corresponding to desired width in temporal domain
    cmw(frq, :) = exp(2 * 1i * pi * tf_freqs(frq) .* wtime) .* exp((-4 * log(2) * wtime.^2) ./ (fwhmTs(frq) / 1000)^2);

    % Normalize wavelet
    cmw(frq, :) = cmw(frq, :) ./ max(cmw(frq, :));

    % Create normalized freq domain wavelet
    cmwX(frq, :) = fft(cmw(frq, :)) ./ max(fft(cmw(frq, :)));

    % Determine observed fwhmT
    midt = dsearchn(wtime', 0);
    cmw_amp = abs(cmw(frq, :)) ./ max(abs(cmw(frq, :))); % Normalize cmw amplitude
    obs_fwhmT(frq) = wtime(midt - 1 + dsearchn(cmw_amp(midt : end)', 0.5)) - wtime(dsearchn(cmw_amp(1 : midt)', 0.5));

    % Determine observed fwhmF
    idx = dsearchn(hz', tf_freqs(frq));
    cmwx_amp = abs(cmwX(frq, :)); 
    obs_fwhmF(frq) = hz(idx - 1 + dsearchn(cmwx_amp(idx : end)', 0.5) - dsearchn(cmwx_amp(1 : idx)', 0.5));

end

% Define time window of analysis
pruned_segs = [-1500, 1000];
tf_times = EEG_TF_INFO.times(dsearchn(EEG_TF_INFO.times', pruned_segs(1)) : dsearchn(EEG_TF_INFO.times', pruned_segs(2)));

% Result matrix
ersp_frontal = zeros(length(subject_list), 8, n_frq, length(tf_times));

% Loop subjects
for s = 1 : length(subject_list)

    % Load subject ERP data
    EEG = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Get between subject condition (1=exp, 2=control)
    group_idx(s) = EEG.trialinfo.session_condition(1);

    % Get trial idx
    idx_close_hi_earl = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr <= 3;
    idx_close_lo_earl = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr <= 3;
    idx_clear_hi_earl = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr <= 3;
    idx_clear_lo_earl = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr <= 3;
    idx_close_hi_late = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr >= 10;
    idx_close_lo_late = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr >= 10;
    idx_clear_hi_late = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr >= 10;
    idx_clear_lo_late = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr >= 10;

    % Channel idx
    chan_idx = [61, 62, 63];

    % Get ERP
    erp_frontal(s, 1, :) = squeeze(mean(EEG.data(chan_idx, :, idx_close_hi_earl), [1, 3]))';
    erp_frontal(s, 2, :) = squeeze(mean(EEG.data(chan_idx, :, idx_close_lo_earl), [1, 3]))';
    erp_frontal(s, 3, :) = squeeze(mean(EEG.data(chan_idx, :, idx_clear_hi_earl), [1, 3]))';
    erp_frontal(s, 4, :) = squeeze(mean(EEG.data(chan_idx, :, idx_clear_lo_earl), [1, 3]))';
    erp_frontal(s, 5, :) = squeeze(mean(EEG.data(chan_idx, :, idx_close_hi_late), [1, 3]))';
    erp_frontal(s, 6, :) = squeeze(mean(EEG.data(chan_idx, :, idx_close_lo_late), [1, 3]))';
    erp_frontal(s, 7, :) = squeeze(mean(EEG.data(chan_idx, :, idx_clear_hi_late), [1, 3]))';
    erp_frontal(s, 8, :) = squeeze(mean(EEG.data(chan_idx, :, idx_clear_lo_late), [1, 3]))';

    % Load subject TF data
    EEG = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');
    d = double(EEG.data);

    % Get trial idx
    idx_close_hi_earl = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr <= 4;
    idx_close_lo_earl = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr <= 4;
    idx_clear_hi_earl = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr <= 4;
    idx_clear_lo_earl = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr <= 4;
    idx_close_hi_late = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr >= 9;
    idx_close_lo_late = EEG.trialinfo.block_wiggleroom == 0 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr >= 9;
    idx_clear_hi_late = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1  & EEG.trialinfo.sequence_nr >= 9;
    idx_clear_lo_late = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1 & EEG.trialinfo.sequence_nr >= 9;

    % tf decomp
    for ch = 1 : length(chan_idx)

        % Talk
        fprintf('\ntf decomp subject %i/%i | chan %i/%i...\n', s, numel(subject_list), ch, length(chan_idx));

        % Pick channel data
        dch = squeeze(d(chan_idx(ch), :, :));

        % Set convolution length
        convlen = size(dch, 1) * size(dch, 2) + size(cmw, 2) - 1;

        % cmw to freq domain and scale
        cmwX = zeros(n_frq, convlen);
        for f = 1 : n_frq
            cmwX(f, :) = fft(cmw(f, :), convlen);
            cmwX(f, :) = cmwX(f, :) ./ max(cmwX(f, :));
        end

        % Get TF-power
        powcube = NaN(n_frq, size(dch, 1), size(dch, 2));
        tmp = fft(reshape(double(dch), 1, []), convlen);
        for f = 1 : n_frq
            as = ifft(cmwX(f, :) .* tmp); 
            as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
            as = reshape(as, size(dch, 1), size(dch, 2));
            powcube(f, :, :) = abs(as) .^ 2;          
        end

        % Cut edge artifacts
        powcube = powcube(:, dsearchn(EEG.times', pruned_segs(1)) : dsearchn(EEG.times', pruned_segs(2)), :);

        % Get condition general baseline values
        ersp_bl = [-1500, -1200];
        tmp = squeeze(mean(powcube, 3));
        [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        blvals = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ersp
        ersp_frontal(s, 1, :, :) = squeeze(ersp_frontal(s, 1, :, :)) + ((10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_close_hi_earl), 3)), blvals))) / length(chan_idx));
        ersp_frontal(s, 2, :, :) = squeeze(ersp_frontal(s, 2, :, :)) + ((10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_close_lo_earl), 3)), blvals))) / length(chan_idx));
        ersp_frontal(s, 3, :, :) = squeeze(ersp_frontal(s, 3, :, :)) + ((10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_clear_hi_earl), 3)), blvals))) / length(chan_idx));
        ersp_frontal(s, 4, :, :) = squeeze(ersp_frontal(s, 4, :, :)) + ((10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_clear_lo_earl), 3)), blvals))) / length(chan_idx));
        ersp_frontal(s, 5, :, :) = squeeze(ersp_frontal(s, 5, :, :)) + ((10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_close_hi_late), 3)), blvals))) / length(chan_idx));
        ersp_frontal(s, 6, :, :) = squeeze(ersp_frontal(s, 6, :, :)) + ((10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_close_lo_late), 3)), blvals))) / length(chan_idx));
        ersp_frontal(s, 7, :, :) = squeeze(ersp_frontal(s, 7, :, :)) + ((10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_clear_hi_late), 3)), blvals))) / length(chan_idx));
        ersp_frontal(s, 8, :, :) = squeeze(ersp_frontal(s, 8, :, :)) + ((10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_clear_lo_late), 3)), blvals))) / length(chan_idx));                        

    end % End chanit

end


ersp_frontal_theta_exp =  squeeze(mean(squeeze(mean(ersp_frontal(group_idx == 1, :, tf_freqs >= 8 & tf_freqs <= 12, :), 1)), 2));
ersp_frontal_theta_cnt =  squeeze(mean(squeeze(mean(ersp_frontal(group_idx == 2, :, tf_freqs >= 8 & tf_freqs <= 12, :), 1)), 2));

figure()
plot(tf_times, ersp_frontal_theta_exp(1, :), 'k:', 'LineWidth', 1.5)
hold on
plot(tf_times, ersp_frontal_theta_exp(2, :), 'b:', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_exp(3, :), 'r:', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_exp(4, :), 'g:', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_exp(5, :), 'k', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_exp(6, :), 'b', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_exp(7, :), 'r', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_exp(8, :), 'g', 'LineWidth', 1.5)
legend({'close hi early', 'close lo early', 'clear hi early', 'clear lo early', 'close hi late', 'close lo late', 'clear hi late', 'clear lo late'})
title('EXP')

figure()
plot(tf_times, ersp_frontal_theta_cnt(1, :), 'k:', 'LineWidth', 1.5)
hold on
plot(tf_times, ersp_frontal_theta_cnt(2, :), 'b:', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_cnt(3, :), 'r:', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_cnt(4, :), 'g:', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_cnt(5, :), 'k', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_cnt(6, :), 'b', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_cnt(7, :), 'r', 'LineWidth', 1.5)
plot(tf_times, ersp_frontal_theta_cnt(8, :), 'g', 'LineWidth', 1.5)
legend({'close hi early', 'close lo early', 'clear hi early', 'clear lo early', 'close hi late', 'close lo late', 'clear hi late', 'clear lo late'})
title('CNT')








aa=bb





ersp_frontal_exp =  squeeze(mean(ersp_frontal(group_idx == 1, :, :, :), 1));
ersp_frontal_cnt =  squeeze(mean(ersp_frontal(group_idx == 2, :, :, :), 1));

% Plot ERSPs
figure()
clims = [-5, 5];
cmap = 'jet';

subplot(2, 4, 1)
pd = squeeze(ersp_frontal_exp(1, :, :));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
colorbar;
title('close hi early', 'FontSize', 10)

subplot(2, 4, 2)
pd = squeeze(ersp_frontal_exp(2, :, :));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
colorbar;
title('close lo early', 'FontSize', 10)

subplot(2, 4, 3)
pd = squeeze(ersp_frontal_exp(3, :, :));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
colorbar;
title('clear hi early', 'FontSize', 10)

subplot(2, 4, 4)
pd = squeeze(ersp_frontal_exp(4, :, :));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
colorbar;
title('clear lo early', 'FontSize', 10)

subplot(2, 4, 5)
pd = squeeze(ersp_frontal_exp(6, :, :));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
colorbar;
title('close hi late', 'FontSize', 10)

subplot(2, 4, 6)
pd = squeeze(ersp_frontal_exp(6, :, :));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
colorbar;
title('close lo late', 'FontSize', 10)

subplot(2, 4, 7)
pd = squeeze(ersp_frontal_exp(7, :, :));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
colorbar;
title('clear hi late', 'FontSize', 10)

subplot(2, 4, 8)
pd = squeeze(ersp_frontal_exp(8, :, :));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
colormap(cmap)
set(gca, 'clim', clims, 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
colorbar;
title('clear lo late', 'FontSize', 10)








erp = squeeze(mean(erp_frontal, 1));

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