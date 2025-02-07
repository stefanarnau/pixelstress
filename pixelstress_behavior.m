clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2024.0/';

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

% Result matrices
res_rt = [];
res_acc = [];

% Loop subjects and calculate condition ERPs
for s = 1 : length(subject_list)

    % Load subject ERP data
    EEG = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

    % Get id
    id = EEG.trialinfo.id(1);

    % Get trial idx
    idx_close = EEG.trialinfo.block_wiggleroom == 0;
    idx_below = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1;
    idx_above = EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1;

    % Iterate blocks
    for bl = 1 : 8

        % get block idx
        block_idx = EEG.trialinfo.block_nr == bl;

        % Get correct idx
        idx_correct = EEG.trialinfo.accuracy == 1 & EEG.trialinfo.sequence_nr > 4;

        % Get rt
        rt = mean(EEG.trialinfo.rt(block_idx & idx_correct));

        % Get accuracy
        acc = sum(idx_correct & block_idx)  / sum(block_idx);

        res_rt(s, bl) = rt;
        res_acc(s, bl) = acc;

    end

    res_rt(s, 9)  = mean(EEG.trialinfo.rt(idx_close & idx_correct));
    res_rt(s, 10) = mean(EEG.trialinfo.rt(idx_below & idx_correct));
    res_rt(s, 11) = mean(EEG.trialinfo.rt(idx_above & idx_correct));

    res_acc(s, 9)  = sum(idx_correct & idx_close)  / sum(idx_close);
    res_acc(s, 10) = sum(idx_correct & idx_below)  / sum(idx_below);
    res_acc(s, 11) = sum(idx_correct & idx_above)  / sum(idx_above);

end

% Plot blocks
figure()
subplot(2, 1, 1)
plot(1:8, res_rt(:, 1 : 8));
hold on
plot(1 : 8, mean(res_rt(:, 1 : 8)), 'k', 'LineWidth', 2)
subplot(2, 1, 2)
plot(1:8, res_acc(:, 1 : 8));

% Plot conditions
figure()
subplot(2, 1, 1)
plot(1:3, res_rt(:, 9 : 11));
hold on
plot(1 : 3, mean(res_rt(:, 9 : 11)), 'k', 'LineWidth', 2)
subplot(2, 1, 2)
plot(1:3, res_acc(:, 9 : 11));