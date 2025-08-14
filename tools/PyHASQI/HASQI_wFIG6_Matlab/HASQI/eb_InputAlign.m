function [xp, yp]=eb_InputAlign(x,y)
% Function to provide approximate temporal alignment of the reference and
% processed output signals. Leading and trailing zeros are then pruned. 
% The function assumes that the two sequences have the same sampling rate: 
% call eb_Resamp24kHz for each sequence first, then call this function to
% align the signals.（同步参考信号和待处理信号）
%
% Calling variables:
% x       input reference sequence
% y       hearing-aid output sequence
%
% Returned values:
% xp   pruned and shifted reference
% yp   pruned and shifted hearing-aid output
%
% James M. Kates, 12 July 2011.

% Match the length of the processed output to the reference for the
% purposes of computing the cross-covariance（互相关）
nx=length(x);
ny=length(y);
nsamp=min(nx,ny); % 匹配信号长度以计算互相关

% Determine the delay of the output relative to the reference
xy=xcov(x(1:nsamp),y(1:nsamp)); %Cross-covariance of the ref and output
[~,index]=max(abs(xy)); %Find the maximum value（返回最大相关项的索引）
delay=nsamp-index; % 确定输出相较于参考信号的延迟

% Back up 2 msec to allow for dispersion ？
fsamp=24000; %Cochlear model input sampling rate in Hz（耳蜗模型输入的采样率/Hz）
delay=delay - 2*fsamp/1000; %Back up 2 msec （人为把y超前）

% Align the output with the reference allowing for the dispersion
if delay > 0
%   Output delayed relative to the reference 输出有延迟
    y=[y(delay+1:ny) zeros(1,delay)]; %Remove the delay（延迟向前移并补零）
else
%   Output advanced relative to the reference 输出超前
    delay=-delay;
    y=[zeros(1,delay) y(1:ny-delay)]; %Add advance （超前值后移并补零）
end

% Find the start and end of the noiseless reference sequence（找参考信号的起始与末尾，修剪信号）
xabs=abs(x);
xmax=max(xabs);
xthr=0.001*xmax; %Zero detection threshold（0检测阈值）
for n=1:nx
%	First value above the threshold working forwards from the beginning
    if xabs(n)>xthr  % 记录从前向后第一个大于阈值的点的标号
		nx0=n;
		break;
    end
end
for n=nx:-1:1
%	First value above the threshold working backwards from the end
	if xabs(n)>xthr % 记录从后向前第一个大于阈值的点
		nx1=n;
		break;
	end
end

% Prune（删除、修剪） the sequences to remove the leading and trailing zeros
if nx1>ny; nx1=ny; end;
xp=x(nx0:nx1);
yp=y(nx0:nx1);



