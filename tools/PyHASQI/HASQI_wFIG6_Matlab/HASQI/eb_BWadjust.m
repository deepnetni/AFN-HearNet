function BW=eb_BWadjust(control,BWmin,BWmax,Level1)
% Function to compute the increase in auditory filter bandwidth in response
% to high signal levels.（高电平时听觉滤波器的带宽增加量）
%
% Calling arguments:
% control     envelope output in the control filter band
% BWmin       auditory filter bandwidth computed for the loss (or NH)
% BWmax       auditory filter bandwidth at maximum OHC damage
% Level1      RMS=1 corresponds to Level1 dB SPL
%
% Returned value:
% BW          filter bandwidth increased for high signal levels
%
% James M. Kates, 21 June 2011.

% Compute the control signal level
cRMS=sqrt(mean(control.^2)); %RMS signal intensity 控制信号的均方根（能量）
% cRMS = cRMS./2e-5;
cdB=20*log10(cRMS) + Level1; %Convert to dB SPL 转为dB SPL的形式

% Adjust the auditory filter bandwidth
if cdB < 50
%   No BW adjustment for the signal below 50 dB SPL 不处理
    BW=BWmin;
elseif cdB > 100
%   Maximum BW if signal above 100 dB SPL 全部采用最大值
    BW=BWmax;
else
%   Linear interpolation between BW at 50 dB and max BW at 100 dB SPL 
    BW=BWmin + ((cdB-50)/50)*(BWmax-BWmin);  % 在最大最小值之间进行线性插值
end




