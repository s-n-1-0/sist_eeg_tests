% 手動で瞬時周波数取得
% 
% instfreq 瞬時周波数の推定(Help)
% https://jp.mathworks.com/help/signal/ref/instfreq.html
% 「条件スペクトル モーメントとしての瞬時周波数と帯域幅」を一部改変

%% チャープ信号を生成
fs = 3e3;
t = 0:1/fs:2;
y = chirp(t,100,1,200,"quadratic");
y = vco(cos(2*pi*t),[0.1 0.4]*fs,fs);

% ---- 手動で瞬時周波数取得(ヒルベルト変換)
spec = fft(y);
spec(3000:6001) = 0;
spec = spec.* 2;
z2 = ifft(spec);
re = real(z2);
im = imag(z2);
iamp = sqrt(re.^2 + im.^2);
phase = atan2(im,re);
phase = unwrap(phase);
ifreq = (fs/(2*pi))*gradient(phase);
% ---- 手動で瞬時周波数取得(STFT経由)
[psPower,psFreq,psTime] = pspectrum(y,fs,'spectrogram');
tfd = psPower;
tfd = tfd/sum(tfd(:));
tfdSum = sum(tfd);
tmp = sum(psFreq.*tfd,1);
psMoment = tmp./tfdSum;

% ---- MALTABの場合
[z,tz] = instfreq(y,fs);
[a,ta] = tfsmoment(y,fs,1,Centralize=false);

%% ---- 以下比較
plot(tz,z,ta,a,'.',t,ifreq,'g',psTime,psMoment,'o');
legend("MATLAB-instfreq","MATLAB-tfsmoment","MY-hilbert","MY-stft");
% 一致していることがわかる