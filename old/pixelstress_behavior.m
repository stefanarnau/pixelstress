clear all;

% PATH VARS
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';
PATH_EEGLAB = '/home/plkn/eeglab2025.0.0/';

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
                '87_1',...
                '7_2',...
                '8_2',...
                '11_2',...
                '12_2',...
                '15_2',...
                '16_2',...
                '20_2',...
                '22_2',...
                '24_2',...
                '27_2',...
                '29_2',...
                '31_2',...
                '33_2',...
                '34_2',...
                '37_2',...
                '39_2',...
                '41_2',...
                '42_2',...
                '45_2',...
                '46_2',...
                '51_2',...
                '53_2',...
                '54_2',...
                '56_2',...
                '59_2',...
                '60_2',...
                '78_2',...
                '80_2',...
               };

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Result matrices
data = [];
counter = 0;

% Loop subjects and calculate condition ERPs
for s = 1 : length(subject_list)

    % Load subject ERP data
    EEG = pop_loadset('filename', ['vp_', subject_list{s}(1 : end - 2), '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

    % Get id
    id = EEG.trialinfo.id(1);

    % Get group
    group = str2num(subject_list{s}(end));

    % Get trajectory idx (close, below, above)
    idx_trajectories = {EEG.trialinfo.block_wiggleroom == 0,...
                        EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == -1,...
                        EEG.trialinfo.block_wiggleroom == 1 & EEG.trialinfo.block_outcome == 1};

    % Iterate trajectories
    for traj = 1 : 3

        % Get correct idx
        idx_correct = EEG.trialinfo.accuracy == 1;

        % Exclude start of block
        idx_nostart = EEG.trialinfo.sequence_nr > 6;

        % Get rt
        rt = mean(EEG.trialinfo.rt( idx_trajectories{traj} & idx_correct & idx_nostart));

        % Get accuracy
        acc = sum(idx_correct & idx_nostart &  idx_trajectories{traj})  / sum( idx_trajectories{traj});

        % Save
        counter = counter + 1;
        data(counter, :) = [id, group, traj, rt, acc];

    end
end

% Add inverse efficiency to matrix
data(:, 6) = data(:, 4) ./ data(:, 5);


% Create a table from your data matrix
tbl = array2table(data, 'VariableNames', {'subject', 'group', 'trajectory', 'rt', 'acc', 'ie'});

% Convert categorical factor to categorical type
tbl.group = categorical(tbl.group);
tbl.trajectory = categorical(tbl.trajectory);

% Fit the mixed linear model rt
lme = fitlme(tbl, 'rt ~ group + trajectory + group*trajectory + (1|subject) + (1|subject:group)');
disp(lme);
anova(lme);
fixedEffects = lme.fixedEffects;
%randomEffects = randomEffects(lme);
plotResiduals(lme);


% Fit the mixed linear model acc
lme = fitlme(tbl, 'acc ~ group + trajectory + group*trajectory + (1|subject) + (1|subject:group)');
disp(lme);
anova(lme);
fixedEffects = lme.fixedEffects;
%randomEffects = randomEffects(lme);
plotResiduals(lme);

% Fit the mixed linear model ie
lme = fitlme(tbl, 'ie ~ group + trajectory + group*trajectory + (1|subject) + (1|subject:group)');
disp(lme);
anova(lme);
fixedEffects = lme.fixedEffects;
%randomEffects = randomEffects(lme);
plotResiduals(lme);


avg_data = groupsummary(tbl, {'trajectory', 'group'}, 'mean', 'rt');


figure;
plot(avg_data.trajectory, avg_data.mean_rt, 'LineWidth', 2, 'Marker', 'o');
xlabel('group');
ylabel('Average Dependent Variable');
title('Average Dependent Variable by Factor 1 and Factor 2');

% Plot
figure;
subplot(1, 3, 1)

% Get unique categories and continuous factor values
categories = unique(tbl.trajectory);
uniqueContFactors = unique(tbl.sequence);

% Initialize variables for storing averaged data
meanResponses = zeros(length(uniqueContFactors), length(categories));
stdResponses = zeros(length(uniqueContFactors), length(categories));

% Calculate mean and standard deviation for each category and continuous factor
for i = 1:length(categories)
    for j = 1:length(uniqueContFactors)
        % Filter data for the current category and continuous factor value
        subset = tbl(tbl.trajectory == categories(i) & tbl.sequence == uniqueContFactors(j), :);
        
        % Compute mean and standard deviation of the response
        meanResponses(j, i) = nanmean(subset.rt);
        stdResponses(j, i) = nanstd(subset.rt);
    end
end

% Create a figure

hold on;

% Plot lines with error bars for each category
for i = 1:length(categories)
    errorbar(uniqueContFactors, meanResponses(:, i), stdResponses(:, i), ...
        'LineWidth', 2, 'DisplayName', char(categories(i)));
end

% Add labels and title
xlabel('Continuous Factor');
ylabel('Response');
title('Response vs. Continuous Factor by Category (Averaged Across Subjects)');

% Add legend
legend('Location', 'best');

% Add grid
grid on;

% Hold off to end the plot
hold off;


subplot(1, 3, 2)

% Get unique categories and continuous factor values
categories = unique(tbl.trajectory);
uniqueContFactors = unique(tbl.sequence);

% Initialize variables for storing averaged data
meanResponses = zeros(length(uniqueContFactors), length(categories));
stdResponses = zeros(length(uniqueContFactors), length(categories));

% Calculate mean and standard deviation for each category and continuous factor
for i = 1:length(categories)
    for j = 1:length(uniqueContFactors)
        % Filter data for the current category and continuous factor value
        subset = tbl(tbl.trajectory == categories(i) & tbl.sequence == uniqueContFactors(j), :);
        
        % Compute mean and standard deviation of the response
        meanResponses(j, i) = nanmean(subset.acc);
        stdResponses(j, i) = nanstd(subset.acc);
    end
end

% Create a figure

hold on;

% Plot lines with error bars for each category
for i = 1:length(categories)
    errorbar(uniqueContFactors, meanResponses(:, i), stdResponses(:, i), ...
        'LineWidth', 2, 'DisplayName', char(categories(i)));
end

% Add labels and title
xlabel('Continuous Factor');
ylabel('Response');
title('Response vs. Continuous Factor by Category (Averaged Across Subjects)');

% Add legend
legend('Location', 'best');

% Add grid
grid on;

% Hold off to end the plot
hold off;


subplot(1, 3, 3)

% Get unique categories and continuous factor values
categories = unique(tbl.trajectory);
uniqueContFactors = unique(tbl.sequence);

% Initialize variables for storing averaged data
meanResponses = zeros(length(uniqueContFactors), length(categories));
stdResponses = zeros(length(uniqueContFactors), length(categories));

% Calculate mean and standard deviation for each category and continuous factor
for i = 1:length(categories)
    for j = 1:length(uniqueContFactors)
        % Filter data for the current category and continuous factor value
        subset = tbl(tbl.trajectory == categories(i) & tbl.sequence == uniqueContFactors(j), :);
        
        % Compute mean and standard deviation of the response
        meanResponses(j, i) = nanmean(subset.ie);
        stdResponses(j, i) = nanstd(subset.ie);
    end
end

% Create a figure

hold on;

% Plot lines with error bars for each category
for i = 1:length(categories)
    errorbar(uniqueContFactors, meanResponses(:, i), stdResponses(:, i), ...
        'LineWidth', 2, 'DisplayName', char(categories(i)));
end

% Add labels and title
xlabel('Continuous Factor');
ylabel('Response');
title('Response vs. Continuous Factor by Category (Averaged Across Subjects)');

% Add legend
legend('Location', 'best');

% Add grid
grid on;

% Hold off to end the plot
hold off;