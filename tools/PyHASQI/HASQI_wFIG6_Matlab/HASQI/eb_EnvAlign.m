function y = eb_EnvAlign(x,y)
% Function to align the envelope of the processed signal to that of the
% reference signal.（将参考信号与处理信号的包络同步）
%
% Calling arguments:
% x      envelope or BM motion of the reference signal（包络或者基底膜运动）
% y      envelope or BM motion of the output signal
%
% Returned values:
% y      shifted output envelope to match the input （将y信号的包络进行移动以实现同步）
%
% James M. Kates, 28 October 2011.
% Absolute value of the cross-correlation peak removed, 22 June 2012.
% Cross-correlation range reduced, 13 August 2013.

% Correlation parameters 相关计算所需的参数
% Reduce the range of the xcorr calculation to save computation time（缩小计算x相关的范围以缩短计算时间）
fsamp=24000; %Sampling rate in Hz
range=100; %Range in msec（ms） for the xcorr calculation
lags=round(0.001*range*fsamp); %Range in samples
npts=length(x);
lags=min(lags,npts); %Use min of lags, length of the sequence

% Cross-correlate the two sequences over the lag range
xy=xcorr(x,y,lags-1);
[~,location]=max(xy); %Find the peak 找到两信号的最大相关位置

% Compute the delay 计算时延
delay=lags - location;

% Time shift the output sequence
if delay > 0
%   Output delayed relative to the reference （输出相较参考信号有延迟）
    y=[y(delay+1:npts); zeros(delay,1)]; %Remove the delay
elseif delay < 0
%   Output advanced relative to the reference （输出超前于参考信号）
    delay=-delay;
    y=[zeros(delay,1); y(1:npts-delay)]; %Add advance 
end

end

