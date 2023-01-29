%% common settings
root_dir = "..\..\..\dataset\lord2\train";
%% settings 1
input_dir = root_dir + "\build\1\*.edf";
snapshot_dir = root_dir + "\snapshot\1";
export_dir = root_dir + "\pre2\1";
markers = {"Marker__dark" "Marker__light"};
%% settings 2
input_dir = root_dir + "\build\2\*.edf";
snapshot_dir = root_dir + "\snapshot\2";
export_dir = root_dir + "\pre2\2";
markers = {"Wait__dark" "Wait__light"}; %#ok<*CLARRSTR> 
%% preprocessing
filelst = dir(input_dir);
sz = size(filelst);
sz = sz(1);
disp(sz);
for j = 1:sz
    f = filelst(j);
    preprocessing_eeg(f.folder + "\" + f.name,export_dir,snapshot_dir,markers)
end
function preprocessing_eeg(full_filepath,export_dir_path,snapshot_dir_path,markers)
    [filepath,filename,ext] = fileparts(full_filepath);
    eeg = pop_biosig({convertStringsToChars(filepath + "\" + filename + ext)});
    %%フィルタリング
    eeg = pop_eegfiltnew(eeg,1,[]);
    eeg = pop_eegfiltnew(eeg,[],30);
    eeg = pop_epoch(eeg,markers,[-1 2]);
    eeg =  pop_saveset(eeg,"filename",convertStringsToChars(snapshot_dir_path + "\" + filename + ".set")); %スナップショット保存
    disp("--")
    is_after_reject = 0;%0だとディスプレイ表示のリジェクトする前と母数が一致
    eeg = pop_eegthresh(eeg,1,[1:10],-100,100,-1,1.998,0,is_after_reject,0); %"Find abnormal values" debug : 200
    eeg = pop_rejtrend(eeg,1,[1:10],1500,0.5,0.3,0,is_after_reject,0);
    eeg = pop_jointprob(eeg,1,[1:10],5,5,0,is_after_reject,0,0,0);
    eeg = pop_rejkurt(eeg,1,[1:10],5,5,0,is_after_reject,0,0,0);
    eeg = pop_rejspec(eeg,1,"threshold",[-60 40],"freqlimits",[0,40],"eegplotreject",is_after_reject);
    if ~is_after_reject
        eeg = eeg_rejsuperpose(eeg, 1,1,1,1,1,1,1,1);
        eeg = pop_rejepoch(eeg,eeg.reject.rejglobal,0);
    end
    pop_saveset(eeg,"filename",convertStringsToChars(export_dir_path + "\" + filename + ".set"));
end

%pop_editset(eeg)
%eegplot(eeg.data,'srate',500) %srate = fs
%pop_rejmenu