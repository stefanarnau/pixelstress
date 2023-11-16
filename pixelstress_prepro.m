clear all;

% PATH VARS
PATH_EEGLAB = '/home/plkn/eeglab2023.1/';
PATH_RAW = '/mnt/data_dump/pixelstress/eeg_raw/';
PATH_CONTROL_FILES = '/home/plkn/repos/pixelstress/control_files/';
PATH_ICSET = '';
PATH_AUTOCLEANED = '';

% Get bdf file list
fl = dir([PATH_RAW, '*.vhdr']);

% Build a list of subject numbers
subject_id = [];
session_condition = [];
for f = 1 : numel(fl)

    % Get fn integers
    int_cell = regexp(fl(f).name,'\d*', 'Match');

    % Save
    subject_id(f) = str2double(int_cell{2});
    session_condition(f) = str2double(int_cell{3});
end

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Get chanlocfile
channel_location_file = which('standard-10-5-cap385.elp');

% Loop subjects
for s = 1 : length(fl)

    % Load EEG
    EEG = pop_loadbv(PATH_RAW, fl(s).name, [], []);

    % Set session string
    if session_condition(s) == 1
        str_cond = 'exp';
    elseif session_condition(s) == 2
        str_cond = 'cntr';
    end

    % Load control file
    CNT = readtable([PATH_CONTROL_FILES, 'control_file_', num2str(subject_id(s)), '_', str_cond, '.csv']);

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
        fprintf('\n\n\nSOMETHING IS WEIIIRDDD!!!!!!\n\n\n');
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
                                                subject_id(s),...
                                                session_condition(s),... 
                                                trial_nr_total,...
                                                block_nr,...
                                                ];

            end
        end
    end

    % Check trialcount
    if trial_nr_total ~= 768
        fprintf('\n\n\nSOMETHING IS WEIIIRDDD!!!!!!\n\n\n');
        pause;
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

    % TODO !!!!!!!!

    aa = bb;


    % Fork response button channels
    EEG = pop_select(EEG, 'channel', [1 : 64]);

    % Add FCz as empty channel
    EEG.data(end + 1, :) = 0;
    EEG.nbchan = size(EEG.data, 1);
    EEG.chanlocs(end + 1).labels = 'FCz';

    % Add channel locations
    EEG = pop_chanedit(EEG, 'lookup', channel_location_file);

    % Save original channel locations (for later interpolation)
    EEG.chanlocs_original = EEG.chanlocs;

    % Reref to CPz, so that FCz obtains non-interpolated data
    EEG = pop_reref(EEG, 'CPz');

    % Resample data
    EEG    = pop_resample(EEG, 200);
    EEG_TF = pop_resample(EEG, 200);

    % Filter
    EEG    = pop_basicfilter(EEG,    [1 : EEG.nbchan],    'Cutoff', [0.01, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 6, 'RemoveDC', 'on', 'Boundary', 'boundary'); 
    EEG_TF = pop_basicfilter(EEG_TF, [1 : EEG_TF.nbchan], 'Cutoff', [   2, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 6, 'RemoveDC', 'on', 'Boundary', 'boundary');
        
    % Bad channel detection
    [EEG, EEG.chans_rejected]       = pop_rejchan(EEG,    'elec', [1 : EEG.nbchan],    'threshold', 5, 'norm', 'on', 'measure', 'kurt');
    [EEG_TF, EEG_TF.chans_rejected] = pop_rejchan(EEG_TF, 'elec', [1 : EEG_TF.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'kurt');

    % Interpolate channels
    EEG    = pop_interp(EEG,    EEG.chanlocs_original,    'spherical');
    EEG_TF = pop_interp(EEG_TF, EEG_TF.chanlocs_original, 'spherical');

    % Reref common average
    EEG    = pop_reref(EEG,    []);
    EEG_TF = pop_reref(EEG_TF, []);

    % Determine rank of data
    dataRank = sum(eig(cov(double(EEG_TF.data'))) > 1e-6); 

    % Epoch EEG data
    EEG    = pop_epoch(EEG, {'X'}, [-0.3, 1.6], 'newname', [subject '_epoched'], 'epochinfo', 'yes');
    EEG    = pop_rmbase(EEG, [-200, 0]);
    EEG_TF = pop_epoch(EEG_TF, {'X'}, [-0.8, 2.5], 'newname', [subject '_epoched'],  'epochinfo', 'yes');
    EEG_TF = pop_rmbase(EEG_TF, [-200, 0]);

    % Autoreject trials
    [EEG,    EEG.rejected_epochs]    = pop_autorej(EEG,    'nogui', 'on');
    [EEG_TF, EEG_TF.rejected_epochs] = pop_autorej(EEG_TF, 'nogui', 'on');

    % Remove from trialinfo
    EEG.trialinfo(EEG.rejected_epochs, :) = [];
    EEG_TF.trialinfo(EEG_TF.rejected_epochs, :) = [];

    % Runica & ICLabel
    EEG_TF = pop_runica(EEG_TF, 'extended', 1, 'interrupt', 'on', 'PCA', dataRank);
    EEG_TF = iclabel(EEG_TF);

    % Find nobrainer
    EEG_TF.nobrainer = find(EEG_TF.etc.ic_classification.ICLabel.classifications(:, 1) < 0.3 | EEG_TF.etc.ic_classification.ICLabel.classifications(:, 3) > 0.3);

    % Copy ICs to erpset
    EEG = pop_editset(EEG, 'icachansind', 'EEG_TF.icachansind', 'icaweights', 'EEG_TF.icaweights', 'icasphere', 'EEG_TF.icasphere');
    EEG.etc = EEG_TF.etc;
    EEG.nobrainer = EEG_TF.nobrainer;

    % Save IC set
    pop_saveset(EEG,    'filename', [subject, '_icset_erp.set'], 'filepath', PATH_ICSET, 'check', 'on');
    pop_saveset(EEG_TF, 'filename', [subject, '_icset_tf.set'],  'filepath', PATH_ICSET, 'check', 'on');

    % Remove components
    EEG    = pop_subcomp(EEG, EEG.nobrainer, 0);
    EEG_TF = pop_subcomp(EEG_TF, EEG_TF.nobrainer, 0);

    % Save clean data
    pop_saveset(EEG, 'filename',    [subject, '_cleaned_erp.set'],       'filepath', PATH_AUTOCLEANED, 'check', 'on');
    pop_saveset(EEG_TF, 'filename', [subject, '_cleaned_tf.set'],        'filepath', PATH_AUTOCLEANED, 'check', 'on');

end % End subject loop


