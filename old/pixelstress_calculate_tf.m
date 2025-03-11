clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';
PATH_TF_DATA = '/mnt/data_dump/pixelstress/3_tf_data/';

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

% Load TF info
EEG_TF_INFO = pop_loadset('filename', ['vp_', subject_list{1}(1 : end - 2), '_cleaned_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Set complex Morlet wavelet parameters
n_frq = 35;
frqrange = [2, 30];
tfres_range = [400, 100];

% Set wavelet time
wtime = -2 : 1 / EEG_TF_INFO.srate : 2;

% Determine fft frqs
hz = logspace(0, EEG_TF_INFO.srate, length(wtime));

% Create wavelet frequencies and tapering Gaussian widths in temporal domain
tf_freqs = logspace(frqrange(1), frqrange(2), n_frq);
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

% Table for between factor
group_idx = [];

% Loop subjects
for s = 1 : length(subject_list)

    % Result matrices
    ersps = zeros(3, EEG_TF_INFO.nbchan, n_frq, length(tf_times));
    itpcs = zeros(3, EEG_TF_INFO.nbchan, n_frq, length(tf_times));

    % Load subject TF data
    EEG = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');
    d = double(EEG.data);

    % Get between subject condition (1=exp, 2=control)
    group_idx(s, :) = [str2double(subject_list{s}(1 : end - 2)), EEG.trialinfo.session_condition(1)];

    % Get trial idx
    idx_close = EEG.trialinfo.sequence_nr >= 7 & EEG.trialinfo.block_wiggleroom == 0;
    idx_below = EEG.trialinfo.sequence_nr >= 7 & EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1;
    idx_above = EEG.trialinfo.sequence_nr >= 7 & EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1 ;

    % tf decomp
    for ch = 1 : EEG.nbchan

        % Talk
        fprintf('\ntf decomp subject %i/%i | chan %i/%i...\n', s, numel(subject_list), ch, EEG.nbchan);

        % Pick channel data
        dch = squeeze(d(ch, :, :));

        % Set convolution length
        convlen = size(dch, 1) * size(dch, 2) + size(cmw, 2) - 1;

        % cmw to freq domain and scale
        cmwX = zeros(n_frq, convlen);
        for f = 1 : n_frq
            cmwX(f, :) = fft(cmw(f, :), convlen);
            cmwX(f, :) = cmwX(f, :) ./ max(cmwX(f, :));
        end

        % Get TF-stuff
        powcube = NaN(n_frq, size(dch, 1), size(dch, 2));
        phacube = NaN(n_frq, size(dch, 1), size(dch, 2));
        tmp = fft(reshape(double(dch), 1, []), convlen);
        for f = 1 : n_frq
            as = ifft(cmwX(f, :) .* tmp); 
            as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
            as = reshape(as, size(dch, 1), size(dch, 2));
            powcube(f, :, :) = abs(as) .^ 2;
            phacube(f, :, :) = angle(as);        
        end

        % Cut edge artifacts
        powcube = powcube(:, dsearchn(EEG.times', pruned_segs(1)) : dsearchn(EEG.times', pruned_segs(2)), :);
        phacube = phacube(:, dsearchn(EEG.times', pruned_segs(1)) : dsearchn(EEG.times', pruned_segs(2)), :);

        % Get condition general baseline values
        ersp_bl = [-1500, -1200];
        tmp = squeeze(mean(powcube, 3));
        [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        blvals = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ersp
        ersps(1, ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_close), 3)), blvals));
        ersps(2, ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_below), 3)), blvals));
        ersps(3, ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_above), 3)), blvals));        
        
        % Calculate itpc
        itpcs(1, ch, :, :) = abs(squeeze(mean(exp(1i*phacube(:, :, idx_close)), 3)));
        itpcs(2, ch, :, :) = abs(squeeze(mean(exp(1i*phacube(:, :, idx_below)), 3)));  
        itpcs(3, ch, :, :) = abs(squeeze(mean(exp(1i*phacube(:, :, idx_above)), 3)));

    end % End chanit

    % Save stuff
    save([PATH_TF_DATA, 'vp_', subject_list{s}(1 : end - 2),'_ersps.mat'], 'ersps');
    save([PATH_TF_DATA, 'vp_', subject_list{s}(1 : end - 2),'_itpcs.mat'], 'itpcs');

end

% Save group info
save([PATH_TF_DATA, 'group_idx.mat'], 'group_idx');

% Save times and freqs and chanlocs
save([PATH_TF_DATA, 'tf_times.mat'], 'tf_times');
save([PATH_TF_DATA, 'tf_freqs.mat'], 'tf_freqs');
chanlocs = EEG.chanlocs;
save([PATH_TF_DATA, 'chanlocs.mat'], 'chanlocs');
