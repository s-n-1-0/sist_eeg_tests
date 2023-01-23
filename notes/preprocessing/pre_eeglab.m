%% 
eeg = pop_biosig({'./ex.edf'});
%%フィルタリング
eeg = pop_eegfiltnew(eeg,1,[]);
eeg = pop_eegfiltnew(eeg,[],40);
eeg = pop_epoch(eeg,{"Marker__dark" "Marker__light"},[-1 2]); %#ok<CLARRSTR> 
disp("--")
is_after_reject = 0;%0だとディスプレイ表示のリジェクトする前と母数が一致
eeg = pop_eegthresh(eeg,1,[1:10],-200,200,-1,1.998,0,is_after_reject,0); %"Find abnormal values" debug : 200
eeg = pop_rejtrend(eeg,1,[1:10],1500,0.5,0.3,0,is_after_reject,0);
eeg = pop_jointprob(eeg,1,[1:10],5,5,0,is_after_reject,0,0,0);
eeg = pop_rejkurt(eeg,1,[1:10],5,5,0,is_after_reject,0,0,0);
eeg = pop_rejspec(eeg,1,"threshold",[-60 40],"freqlimits",[0,40],"eegplotreject",is_after_reject);
if ~is_after_reject
    eeg = eeg_rejsuperpose(eeg, 1,1,1,1,1,1,1,1);
    eeg = pop_rejepoch(eeg,eeg.reject.rejglobal,0);
end
%% 
pop_saveset(eeg,"filename",'./ex.set');

%pop_editset(eeg)
%eegplot(eeg.data,'srate',500) %srate = fs
%pop_rejmenu