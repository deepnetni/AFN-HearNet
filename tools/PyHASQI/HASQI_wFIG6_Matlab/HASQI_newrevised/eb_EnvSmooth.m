function [smooth,FBE]=eb_EnvSmooth(env,segsize,fsamp)
% Function to smooth the envelope returned by the cochlear model. The
% envelope is divided into segments having a 50% overlap��50%�ص���. 
% Each segment is  windowed, summed, and divided by the window sum to 
% produce the average.
% A raised cosine window�������Ҵ��� is used. 
% The envelope sub-sampling frequency is 2*(1000/segsize).
%
% Calling arguments:
% env			matrix of envelopes in each of the auditory bands �������
% segsize		averaging segment size in msec ֡��
% fsamp			input envelope sampling rate in Hz
%
% Returned values:
% smooth		matrix of subsampled windowed averages in each band
% ��Ƶ�����Ӵ��������ƽ������
%
% James M. Kates, 26 January, 2007.
% Final half segment added 27 August 2012.

% Compute the window
nwin=round(segsize*(0.001*fsamp)); %Segment size in samples �����εĴ�С
test=nwin - 2*floor(nwin/2); %0=even, 1=odd
if test>0; nwin=nwin+1; end; %Force window length to be even ǿ�ƴ���Ϊż��
window=hann(nwin); %Raised cosine von Hann window
wsum=sum(window); %Sum for normalization

% The first segment has a half window ��һ֡�а������
nhalf=nwin/2;
halfwindow=window(nhalf+1:nwin);
halfsum=sum(halfwindow);

% Number of segments��֡�ĸ����� and assign the matrix storage
nchan=size(env,1); % env������=�˲������ͨ����
npts=size(env,2); % env������=ÿһͨ��������Ƶ����
nseg=1 + floor(npts/nwin) + floor((npts-nwin/2)/nwin); % ����ÿ��ͨ����֡�ĸ���
smooth=zeros(nchan,nseg);
FBE=smooth;

% Loop to compute the envelope in each frequency band ��ÿ��Ƶ�������������
for k=1:nchan % ����ͨ��
%	Extract the envelope in the frequency band ��ȡ����
	r=env(k,:);

%	The first (half) windowed segment ��һ�Σ����ص��İ�����֣�
	nstart=1;
    wr=r(nstart:nhalf).*halfwindow';
	smooth(k,1)=sum(r(nstart:nhalf).*halfwindow')/halfsum;
    FBE(k,1)=sum(10.^(wr/20))*2;

%	Loop over the remaining full segments, 50% overlap ��ʣ�µķֶα���������50%��֡�ص�
    for n=2:nseg-1
		nstart=nstart + nhalf;
		nstop=nstart + nwin - 1;
		wr=r(nstart:nstop).*window';
		smooth(k,n)=sum(wr)/wsum;
        FBE(k,n)=sum(10.^(wr/20));
    end
    
%   The last (half) windowed segment ���һ�Σ����ص��İ�����֣�
 	nstart=nstart + nhalf;
	nstop=nstart + nhalf - 1;
    wr=r(nstart:nstop).*window(1:nhalf)';
    smooth(k,nseg)=sum(wr)/halfsum;        
    FBE(k,nseg)=sum(10.^(wr/20))*2;   
end
