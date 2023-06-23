%% 
filelst = dir("MIOnly_FTP_EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms\session1");
fs = 1000;
filelst = filelst(endsWith({filelst.name},"MI.mat"));
sz = size(filelst);
disp(sz(1));

%% preview
fp = filelst(2);
data = extract_data(fp.folder+"/"+fp.name);
preprocessing_eeg(fp.folder + "/pre/"+fp.name + ".set",data{1},fs)
%% 
for i = 1:sz
    f = filelst(i);
    disp(f.name)
    data = extract_data(f.folder+"/"+f.name);
    preprocessing_eeg(f.folder + "/pre/"+f.name + ".set",data{1},fs)
end
function preprocessing_eeg(export_dir_path,data,fs)
    eegdata = data{1}.';%%ch × samples
    eegindexes = data{2};

    eeg = pop_importdata('dataformat', 'array', 'data', eegdata, 'setname', 'EEG', 'srate', fs);
    % イベントを追加する
    eeg = eeg_addnewevents(eeg,{eegindexes},{'x'});
    % イベント構造を更新する
    %eeg = eeg_checkset(eeg);
    %%ダウンサンプリング
    eeg = pop_resample(eeg,250);
    %%フィルタリング
    eeg = pop_eegfiltnew(eeg,1,[]);
    eeg = pop_eegfiltnew(eeg,[],30);
    eeg = pop_epoch(eeg, {"x"}, [0, 4]);
    is_after_reject = 0;%0だとディスプレイ表示のリジェクトする前と母数が一致
    %TODO:一時的に範囲を上げてる
    eeg = pop_eegthresh(eeg,1,[1:62],-500,500,-1,1.998,0,is_after_reject,0); %"Find abnormal values" default : 100
    eeg = pop_rejtrend(eeg,1,[1:62],1500,0.5,0.3,0,is_after_reject,0);
    eeg = pop_jointprob(eeg,1,[1:62],5,5,0,is_after_reject,0,0,0);
    eeg = pop_rejkurt(eeg,1,[1:62],5,5,0,is_after_reject,0,0,0);
    eeg = pop_rejspec(eeg,1,"threshold",[-60 40],"freqlimits",[0,40],"eegplotreject",is_after_reject);
    if ~is_after_reject
        eeg = eeg_rejsuperpose(eeg, 1,1,1,1,1,1,1,1);
        eeg = pop_rejepoch(eeg,eeg.reject.rejglobal,0);
    end
    pop_saveset(eeg,"filename",convertStringsToChars(export_dir_path));
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