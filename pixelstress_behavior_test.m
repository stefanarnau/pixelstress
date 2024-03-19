clear all;

% PATH VARS
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';
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
               };

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Get chanlocfile
channel_location_file = which('standard-10-5-cap385.elp');

res = [];

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

    CNT = renamevars(CNT, ["block_nr"], ["block_nr_cnt"]);

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

    % Check trialcount
    if trial_nr_total ~= 768
        fprintf('\n\n\nSOMETHING IS WEIIIRDDD with the trials!!!!!!\n\n\n');
        pause;
    end

    % Convert trialinfo to table
    trialinfo = array2table(trialinfo, 'VariableNames', {'event_number', 'id', 'session_condition', 'trial_nr_total', 'block_nr'});

    % Combine info
    trialinfo = [trialinfo, CNT(:, [6 : end])];

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


    % Accuracy in easy trials
    acc = sum(trialinfo.accuracy == 1) / size(trialinfo, 1);

    res(s, :) = [trialinfo.id(1), acc];

end % End subject loop


