clear all;

% PATH VARS
PATH_EEGLAB = '/home/plkn/eeglab2022.1/';
PATH_RAW = '/mnt/data_dump/pixelstress/eeg_raw/';
PATH_ICSET = '';
PATH_AUTOCLEANED = '';

% Subject list (stating the obvious here...)
subject_list = {'ExpExp104_2_2_1'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Get chanlocfile
channel_location_file = which('standard-10-5-cap385.elp');

% Loop subjects
for s = 1 : length(subject_list)

    % Get id stuff
    subject = subject_list{s};

    % Load
    EEG = pop_loadbv(PATH_RAW, [subject, '.vhdr'], [], []);

    % Iterate events
    trialinfo = [];
    block_nr = -1;
    trial_nr = 0;
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

            % If trial
            if enum == 100

                % Increase!!!!!
                trial_nr = trial_nr + 1;

                % Save info
                trialinfo(trial_nr, :) = [trial_nr, block_nr];


            end

        end

    end

    enums= horzcat([1:256]', enums);

    enums(enums(:, 2)  == 0, :) = [];

    aa=bb

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
    EEG    = pop_epoch(EEG, {'X'}, [-0.3, 2], 'newname', [subject '_epoched'], 'epochinfo', 'yes');
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
    pop_saveset(EYE, 'filename',    [subject, '_cleaned_eye_erp.set'],   'filepath', PATH_AUTOCLEANED, 'check', 'on');
    pop_saveset(EYE_TF, 'filename', [subject, '_cleaned_eye_tf.set'],    'filepath', PATH_AUTOCLEANED, 'check', 'on');

end % End subject loop


