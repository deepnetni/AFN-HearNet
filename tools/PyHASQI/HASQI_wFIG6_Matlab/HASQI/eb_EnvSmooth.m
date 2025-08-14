function [smooth,FBE]=eb_EnvSmooth(env,segsize,fsamp)
% Function to smooth the envelope returned by the cochlear model. The
% envelope is divided into segments having a 50% overlap（50%重叠）. 
% Each segment is  windowed, summed, and divided by the window sum to 
% produce the average.
% A raised cosine window（升余弦窗） is used. 
% The envelope sub-sampling frequency is 2*(1000/segsize).
%
% Calling arguments:
% env			matrix of envelopes in each of the auditory bands 包络矩阵
% segsize		averaging segment size in msec 帧长
% fsamp			input envelope sampling rate in Hz
%
% Returned values:
% smooth		matrix of subsampled windowed averages in each band
% 各频带经加窗采样后的平均矩阵
%
% James M. Kates, 26 January, 2007.
% Final half segment added 27 August 2012.

% Compute the window
nwin=round(segsize*(0.001*fsamp)); %Segment size in samples 样本段的大小
test=nwin - 2*floor(nwin/2); %0=even, 1=odd
if test>0; nwin=nwin+1; end; %Force window length to be even 强制窗长为偶数
window=hann(nwin); %Raised cosine von Hann window
wsum=sum(window); %Sum for normalization

% The first segment has a half window 第一帧有半个窗口
nhalf=nwin/2;
halfwindow=window(nhalf+1:nwin);
halfsum=sum(halfwindow);

% Number of segments（帧的个数） and assign the matrix storage
nchan=size(env,1); % env的行数=滤波器组的通道数
npts=size(env,2); % env的列数=每一通道的中心频率数
nseg=1 + floor(npts/nwin) + floor((npts-nwin/2)/nwin); % 计算每个通道内帧的个数
smooth=zeros(nchan,nseg);
FBE=smooth;

% Loop to compute the envelope in each frequency band 在每个频带遍历计算包络
for k=1:nchan % 遍历通道
%	Extract the envelope in the frequency band 提取包络
	r=env(k,:);

%	The first (half) windowed segment 第一段（不重叠的半个部分）
	nstart=1;
    wr=r(nstart:nhalf).*halfwindow';
	smooth(k,1)=sum(r(nstart:nhalf).*halfwindow')/halfsum;
    FBE(k,1)=sum(10.^(wr/20))*2;

%	Loop over the remaining full segments, 50% overlap 对剩下的分段遍历，并有50%的帧重叠
    for n=2:nseg-1
		nstart=nstart + nhalf;
		nstop=nstart + nwin - 1;
		wr=r(nstart:nstop).*window';
		smooth(k,n)=sum(wr)/wsum;
        FBE(k,n)=sum(10.^(wr/20));
    end
    
%   The last (half) windowed segment 最后一段（不重叠的半个部分）
 	nstart=nstart + nhalf;
	nstop=nstart + nhalf - 1;
    wr=r(nstart:nstop).*window(1:nhalf)';
    smooth(k,nseg)=sum(wr)/halfsum;        
    FBE(k,nseg)=sum(10.^(wr/20))*2;   
end
