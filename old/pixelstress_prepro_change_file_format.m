
fid = fopen('chanlabels_pixelstress.txt', 'w');
for ch = 1 : length(EEG.chanlocs)

    chlab = EEG.chanlocs(ch).labels;

    fprintf(fid, '%s\n', chlab);

end
fclose(fid);