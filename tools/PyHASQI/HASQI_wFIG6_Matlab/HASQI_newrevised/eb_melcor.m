function [m1,xy,vad]=eb_melcor(x,y,thr,addnoise)
% Function to compute the cross-correlations between the input signal
% time-frequency envelope（输入信号的时频包络） and the distortion time-frequency envelope(失真信号的时频包络).
% For each time interval, the log spectrum（对数谱） is fitted with a set of half-cosine
% basis functions（半余弦基函数）. The spectrum weighted by the basis functions（由基函数加权的谱） corresponds
% to（对应于） mel cepstral coefficients computed in the frequency domain（在频域中计算的Mel倒谱系数）. The 
% amplitude-normalized（幅度归一化） cross-covariance（互协方差） between the time-varying basis
% functions for the input and output signals is then computed.
%
% Calling variables:
% x		    subsampled input signal envelope in dB SL in each critical band
% y		    subsampled distorted output signal envelope
% thr	    threshold in dB SPL to include segment in calculation
% addnoise  additive Gaussian noise to ensure 0 cross-corr at low levels（保证低水平下的零互相关）
%
% Output:
% m1		average cepstral correlation 2-6, input vs output
% xy		individual cepstral correlations, input vs output
%
% James M. Kates, 24 October 2006.
% Difference signal removed for cochlear model, 31 January 2007.
% Absolute value added 13 May 2011.
% Changed to loudness criterion for silence threhsold, 28 August 2012.

% Processing parameters
nbands=size(x,1); % 信号矩阵的行数，通道数（频带数）

% Mel cepstrum basis functions （Mel倒谱的基函数）(mel cepstrum because of auditory bands)
nbasis=6; %Number of cepstral coefficients to be used 使用到的倒谱系数的个数
freq=0:nbasis-1;
k=0:nbands-1;
cepm=zeros(nbands,nbasis);
for nb=1:nbasis
	basis=cos(freq(nb)*pi*k/(nbands-1));
	cepm(:,nb)=basis/norm(basis);
end

% Find the segments that lie sufficiently above the quiescent rate
% 找出高静音阈值的段
xLinear=10.^(x/20); %Convert envelope dB to linear (specific loudness) 将dB表示的包络转换为线性
xsum=sum(xLinear,1)/nbands; %Proportional to loudness in sones（宋） 与响度成正比
xsum=20*log10(xsum); %Convert back to dB (loudness in phons) 音高
index=find(xsum > thr); %Identify those segments above threshold 
nsamp=length(index); %Number of segments above threshold

vad=zeros(length(xsum),1);
vad(index)=1;

% Exit if not enough segments above zero 如果所有的段都是静音段（低于静音阈值）
if nsamp <= 1
	m1=0;
	xy=zeros(nbasis,1);
	fprintf('Function eb_melcor: Signal below threshold, outputs set to 0.\n');
	return;
end

% Remove the silent intervals 仅保留高于静音阈值的段
x=x(:,index);
y=y(:,index);

% Add the low-level noise to the envelopes 加入低阶随机白噪声
x=x + addnoise*randn(size(x));
y=y + addnoise*randn(size(y));

% Compute the mel cepstrum coefficients using only those segments
% above threshold 使用有声段计算mel倒谱系数
xcep=zeros(nbasis,nsamp); %Input（倒谱系数的个数，有声段的个数）
ycep=zeros(nbasis,nsamp); %Output
for n=1:nsamp % 遍历有声段
	for k=1:nbasis % 计算倒谱系数
		xcep(k,n)=sum(x(:,n).*cepm(:,k));
		ycep(k,n)=sum(y(:,n).*cepm(:,k));
	end
end

% Remove the average value from the cepstral coefficients.从倒谱系数中去掉平均值
% The cross-correlation（互相关） thus becomes a cross-covariance（互协方差）, and there
% is no effect of the absolute signal level in dB.
for k=1:nbasis
	xcep(k,:)=xcep(k,:) - mean(xcep(k,:));
	ycep(k,:)=ycep(k,:) - mean(ycep(k,:));
end

% Normalized cross-correlations between the time-varying cepstral coeff（时变倒谱系数的归一化互相关）
xy=zeros(nbasis,1); %Input vs output
small=1.0e-30;
for k=1:nbasis
	xsum=sum(xcep(k,:).^2); % Mel倒谱系数的平方和
	ysum=sum(ycep(k,:).^2);
	if (xsum < small) || (ysum < small)
		xy(k)=0.0;
	else
		xy(k)=abs(sum(xcep(k,:).*ycep(k,:))/sqrt(xsum*ysum));
	end
end

% Figure of merit is the average of the cepstral correlations, ignoring
% the first (average spectrum level).
m1=sum(xy(2:nbasis))/(nbasis-1); % 倒谱系数的归一化互相关的平均值（去除第一个值？）
