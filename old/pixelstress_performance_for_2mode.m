clear all;

% PATH VARS
PATH_EEGLAB = '/home/plkn/eeglab2024.0/';
PATH_RAW = '/mnt/data_dump/pixelstress/0_eeg_raw/';
PATH_CONTROL_FILES = '/mnt/data_dump/pixelstress/0_control_files/';
PATH_ICSET = '/mnt/data_dump/pixelstress/1_icset/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelstress/2_autocleaned/';


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
                '18_2',...
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
                '43_1',...
                '44_1',...
                '45_2',...
                '46_2',...
                '47_1',...
                '48_1',...
                '50_1',...
                '51_2',...
                '52_1',...
                '53_2',...
                '54_2',...
                '55_1',...
                '56_2',...
                '57_1',...
                '58_1',...
                '59_2',...
                '60_2',...
               };

% Failed:
% '13_1' trialcount too low. Notes: Abbruch wegen Kreislaufproblemen.
% '49_2' triggers stopped in block 7

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Get chanlocfile
channel_location_file = which('standard-10-5-cap385.elp');

% Init Outdata
outdata = [];

% Loop subjects
for s = 1 : length(subject_list)

    % Load EEG
    fn = ['ExpExp104_', subject_list{s}, '_1.vhdr'];
    EEG = pop_loadbv(PATH_RAW, fn, [], []);

    % Set session string
    if str2num(subject_list{s}(end)) == 1
        str_cond = 'exp';
    elseif str2num(subject_list{s}(end)) == 2
        str_cond = 'cntr';
    end

    % Get id
    subject_id = str2num(subject_list{s}(1 : end - 2));

    % Load control file
    CNT = readtable([PATH_CONTROL_FILES, 'control_file_', num2str(subject_id), '_', str_cond, '.csv']);

    % Copy feedback info to following sequence trials
    fb = NaN;
    fb_scaled = NaN;
    for e = 1 : size(CNT, 1)
        if CNT(e, :).event_code == 4
            fb = CNT(e, :).sequence_feedback;
            fb_scaled = CNT(e, :).sequence_feedback_scaled;
        end
        if CNT(e, :).event_code == 5
            CNT(e, :).sequence_feedback = fb;
            CNT(e, :).sequence_feedback_scaled = fb_scaled;

            % Set feedback NaN if first sequence of block
            if CNT(e, :).sequence_nr == 1
                CNT(e, :).sequence_feedback = NaN;
                CNT(e, :).sequence_feedback_scaled = NaN;
            end
        end
    end

    % Drop non-trial lines
    CNT = CNT(CNT.event_code == 5, :);

    % Check trialcount
    if size(CNT, 1) ~= 768
        fprintf('\n\n\nSOMETHING IS WEIIIRDDD with control file things!!!!!!\n\n\n');
        pause;
    end

    % Iterate events
    trialinfo = [];
    block_nr = -1;
    trial_nr_total = 0;
    enums = zeros(256, 1);
    for e = 1 : length(EEG.event)

        % If an S event
        if strcmpi(EEG.event(e).type(1), 'S')

            % Get event number
            enum = str2num(EEG.event(e).type(2 : end));

            enums(enum) = enums(enum) + 1;

            % Set block number
            if enum >= 210 & enum <= 220
                block_nr = str2num(EEG.event(e).type(end));
            end

            % If trial and no practice block
            if enum == 100 & block_nr > 0

                % Save info
                trial_nr_total = trial_nr_total + 1;
                trialinfo(trial_nr_total, :) = [e,...
                                                subject_id,...
                                                str2num(subject_list{s}(end)),... 
                                                trial_nr_total,...
                                                block_nr,...
                                                ];

            end
        end
    end

    % Trigger recording stopped in block 7 for subject 49_2 after trial 672
    if strcmpi(subject_list{s}, '49_2')
        CNT = CNT(1 : 672, :);
    else

        % Check trialcount
        if trial_nr_total ~= 768
            fprintf('\n\n\nSOMETHING IS WEIIIRDDD with the trials!!!!!!\n\n\n');
            pause;
        end
    end

    % Convert trialinfo to table
    trialinfo = array2table(trialinfo, 'VariableNames', {'event_number', 'id', 'session_condition', 'trial_nr_total', 'block_nr'});

    % Combine info
    trialinfo = [trialinfo, CNT(:, [7 : end])];

    % Rename vars for clarity
    trialinfo = renamevars(trialinfo, ["sequence_feedback", "sequence_feedback_scaled"], ["last_feedback", "last_feedback_scaled"]);

    % Mark trials in event structure
    for t = 1 : size(trialinfo, 1)
        EEG.event(trialinfo(t, :).event_number).code = 'X';
        EEG.event(trialinfo(t, :).event_number).type = 'X';
    end

    % Detect responses
    response_data = [];
    response_counter = 0;
    for e = 1 : length(EEG.event)

        % If trial
        if strcmpi(EEG.event(e).type, 'X')

            % Loop for response
            resp = 0;
            resp_lat = 0;
            f = e;
            while resp == 0 & resp_lat <= 1200

                f = f + 1;

                % get event latency
                resp_lat = EEG.event(f).latency - EEG.event(e).latency;

                % If response key pressed
                if strcmpi(EEG.event(f).type, 'L  1') & resp_lat <= 1200
                    resp = 1;
                elseif strcmpi(EEG.event(f).type, 'R  1') & resp_lat <= 1200
                    resp = 2;
                end
            end

            % Save to matrix
            response_counter = response_counter + 1;
            response_data(response_counter, :) = [resp, resp_lat, 0];
        end
    end

    % Check trialcount
    if response_counter ~= 768
        fprintf('\n\n\nSOMETHING IS WEIIIRDDD!!!!!!\n\n\n');
        pause;
    end

    % Convert response data to table
    response_data = array2table(response_data, 'VariableNames', {'response_key', 'rt', 'accuracy'});

    % Combine info
    trialinfo = [trialinfo, response_data];

    % Code accuray
    for t = 1 : size(trialinfo, 1)
        if trialinfo(t, :).correct_response == 4 & trialinfo(t, :).response_key == 2
            trialinfo(t, :).accuracy = 1; % correct
        elseif trialinfo(t, :).correct_response == 6 & trialinfo(t, :).response_key == 1
            trialinfo(t, :).accuracy = 1; % correct
        elseif trialinfo(t, :).response_key == 0
            trialinfo(t, :).accuracy = 2; % omission
        else
            trialinfo(t, :).accuracy = 0; % incorrect
        end
    end

    % Init row
    row = [subject_id];

    % Iterate blocks
    for bl = 1 : 8

        % Get subsets
        tinf_early = trialinfo(trialinfo.block_nr == bl & trialinfo.sequence_nr <= 4, :);
        tinf_late  = trialinfo(trialinfo.block_nr == bl & trialinfo.sequence_nr >= 9, :);

        % Get rts
        rt_early = mean(tinf_early(tinf_early.accuracy == 1, :).rt);
        rt_late  = mean(tinf_late(tinf_late.accuracy == 1, :).rt);

        % Get accuracies
        acc_early = size(tinf_early(tinf_early.accuracy == 1, :), 1) / size(tinf_early, 1);
        acc_late = size(tinf_late(tinf_late.accuracy == 1, :), 1) / size(tinf_late, 1);

        % Append
        row = horzcat(row, [rt_early, rt_late, acc_early, acc_late]);

    end

    % collect
    if isempty(outdata)
        outdata = row;
    else
        outdata = vertcat(outdata, row);
    end

    % Convert to table
    outdata_table = array2table(outdata);

    % The colnames
    colnames = {...
        'id',...
        'bl1_rt_early',...
        'bl1_rt_late',...
        'bl1_acc_early',...
        'bl1_acc_late',...
        'bl2_rt_early',...
        'bl2_rt_late',...
        'bl2_acc_early',...
        'bl2_acc_late',...
        'bl3_rt_early',...
        'bl3_rt_late',...
        'bl3_acc_early',...
        'bl3_acc_late',...
        'bl4_rt_early',...
        'bl4_rt_late',...
        'bl4_acc_early',...
        'bl4_acc_late',...
        'bl5_rt_early',...
        'bl5_rt_late',...
        'bl5_acc_early',...
        'bl5_acc_late',...
        'bl6_rt_early',...
        'bl6_rt_late',...
        'bl6_acc_early',...
        'bl6_acc_late',...
        'bl7_rt_early',...
        'bl7_rt_late',...
        'bl7_acc_early',...
        'bl7_acc_late',...
        'bl8_rt_early',...
        'bl8_rt_late',...
        'bl8_acc_early',...
        'bl8_acc_late',...
    };

    % Set colnames
    outdata_table.Properties.VariableNames = colnames;

    % Save to csv
    writetable(outdata_table, 'pxstress_2mode_behavior.csv');

end % End subject loop


