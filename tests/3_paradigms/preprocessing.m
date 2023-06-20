%% 
filelst = dir("MIOnly_FTP_EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms\session1");
filelst = filelst(endsWith({filelst.name},".mat"));
sz = size(filelst);
disp(sz(1));

%%
fp = filelst(2);
data = extract_data(fp.folder+"/"+fp.name);
%% 
for i = 1:sz
    f = filelst(i);
    preprocessing_eeg(f.folder + "\" + f.name,".\pre")
end
function preprocessing_eeg(full_filepath,export_dir_path)
    [filepath,filename,ext] = fileparts(full_filepath);
    eeg = pop_biosig({convertStringsToChars(filepath + "\" + filename + ext)});
    %%フィルタリング
    eeg = pop_eegfiltnew(eeg,1,[]);
    eeg = pop_eegfiltnew(eeg,[],30);
    eeg = pop_epoch(eeg,{"Marker__dark" "Marker__light"},[-1 2]); %#ok<CLARRSTR> 
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

%%
function [ret_train_test_list] = extract_data(path)
    data = load(path);
    train_test_list = [data.EEG_MI_train,data.EEG_MI_test];
    ret_train_test_list = cell(1,2);
    for i = 1:2
        td = train_test_list(i);
        ret_train_test_list{i} = {td.x,td.t};
    end
end
%pop_editset(eeg)
%eegplot(eeg.data,'srate',500) %srate = fs
%pop_rejmenu