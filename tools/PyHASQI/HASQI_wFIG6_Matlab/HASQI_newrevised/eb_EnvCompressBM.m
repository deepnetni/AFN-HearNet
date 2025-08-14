function [y,b]=eb_EnvCompressBM(envsig,bm,control,attnOHC,thrLow,CR,fsamp,Level1)
% Function to compute the cochlear compression in one auditory filter
% band.（实现听觉滤波器的耳蜗压缩）
% The gain is linear below the lower threshold, compressive with
% a compression ratio of CR:1 between the lower and upper thresholds,
% and reverts to linear above the upper threshold. The compressor
% assumes that auditory threshold is 0 dB SPL.（线性-压缩-线性）
%
% Calling variables:
% envsig	analytic signal envelope (magnitude) （解析信号包络）returned by the 
%			gammatone filter bank
% bm        BM motion output by the filter bank （滤波器组输出的基底膜运动）
% control	analytic control envelope（解析控制包络） returned by the wide control
%			path filter bank
% attnOHC	OHC attenuation at the input to the compressor（压缩器输入端的OHC衰减）
% thrLow	kneepoint for the low-level linear amplification （低阶线性放大的拐点）
% CR		compression ratio
% fsamp		sampling rate in Hz
% Level1	dB reference level: a signal having an RMS value of 1 is
%			assigned to Level1 dB SPL.
%
% Function outputs:
% y			compressed version of the signal envelope （压缩后的信号包络）
% b         compressed version of the BM motion （压缩后的基底膜运动）
%
% James M. Kates, 19 January 2007.
% LP filter added 15 Feb 2007 (Ref: Zhang et al., 2001)
% Version to compress the envelope, 20 Feb 2007.
% Change in the OHC I/O function, 9 March 2007.
% Two-tone suppression added 22 August 2008.

% Initialize the compression parameters（初始化压缩参数）
thrHigh=100.0; %Upper compression threshold （高阈值）

% Convert the control envelope to dB SPL（将包络的能量转换dB SPL的形式）
small=1.0e-30;
logenv=max(control,small); %Don't want to take logarithm of zero or neg（防止零或负数的对数）
logenv = logenv./5e-4;
logenv=Level1 + 20*log10(logenv);   % 直接送去计算对数？不用和参考比较一下吗
% 筛选出在高低门限之间的对数包络
logenv=min(logenv,thrHigh); %Clip signal levels above the upper threshold
logenv=max(logenv,thrLow); %Clip signal at the lower threshold

% Compute the compression gain in dB 计算对数增益
gain=-attnOHC - (logenv - thrLow)*(1 - (1/CR));

% Convert the gain to linear and apply a LP filter to give a 0.2 msec delay
gain=10.^(gain/20); % 将增益转换为线性
flp=800;
[b,a]=butter(1,flp/(0.5*fsamp));  % 使用截止频率为800Hz的低通巴特沃斯滤波器
gain=filter(b,a,gain); % 滤波后产生0.2ms的时延

% Apply the gain to the signals
y=gain.*envsig; %Linear envelope
b=gain.*bm; %BM motion
