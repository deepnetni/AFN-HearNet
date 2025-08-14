function y = eb_EnvAlign(x,y)
% Function to align the envelope of the processed signal to that of the
% reference signal.�����ο��ź��봦���źŵİ���ͬ����
%
% Calling arguments:
% x      envelope or BM motion of the reference signal��������߻���Ĥ�˶���
% y      envelope or BM motion of the output signal
%
% Returned values:
% y      shifted output envelope to match the input ����y�źŵİ�������ƶ���ʵ��ͬ����
%
% James M. Kates, 28 October 2011.
% Absolute value of the cross-correlation peak removed, 22 June 2012.
% Cross-correlation range reduced, 13 August 2013.

% Correlation parameters ��ؼ�������Ĳ���
% Reduce the range of the xcorr calculation to save computation time����С����x��صķ�Χ�����̼���ʱ�䣩
fsamp=24000; %Sampling rate in Hz
range=100; %Range in msec��ms�� for the xcorr calculation
lags=round(0.001*range*fsamp); %Range in samples
npts=length(x);
lags=min(lags,npts); %Use min of lags, length of the sequence

% Cross-correlate the two sequences over the lag range
xy=xcorr(x,y,lags-1);
[~,location]=max(xy); %Find the peak �ҵ����źŵ�������λ��

% Compute the delay ����ʱ��
delay=lags - location;

% Time shift the output sequence
if delay > 0
%   Output delayed relative to the reference �������ϲο��ź����ӳ٣�
    y=[y(delay+1:npts); zeros(delay,1)]; %Remove the delay
elseif delay < 0
%   Output advanced relative to the reference �������ǰ�ڲο��źţ�
    delay=-delay;
    y=[zeros(delay,1); y(1:npts-delay)]; %Add advance 
end

end

