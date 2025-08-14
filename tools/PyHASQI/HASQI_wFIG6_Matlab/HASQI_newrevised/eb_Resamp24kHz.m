function [y,fsamp]=eb_Resamp24kHz(x,fsampx)
% Function to resample the input signal at 24 kHz. The input sampling rate
% is rounded to （四舍五入至）the nearest kHz to comput the sampling rate conversion
% ratio.
%
% Calling variables:
% x         input signal
% fsampx    sampling rate for the input in Hz
%
% Returned argument:
% y         signal resampled at 24 kHz
% fsamp     output sampling rate in Kz
%
% James M. Kates, 20 June 2011.

% Sampling rate information
fsamp=24000; %Output sampling rate in Hz（输出信号的采样率）
fy=round(fsamp/1000); %Ouptut rate in kHz （转换单位至kHz）
fx=round(fsampx/1000); %Input rate to nearest kHz

% Resample the signal
if fx == fy
%   No resampling performed if the rates match
    y=x;

elseif fx < fy
%   Resample for the input rate lower than the output
    y=resample(x,fy,fx);
    
%   Match the RMS level of the resampled signal to that of the input
    xRMS=sqrt(mean(x.^2)); % 输入信号的均方根值
    yRMS=sqrt(mean(y.^2)); % 重采样后信号的均方根值
    y=(xRMS/yRMS)*y; % 二者相匹配，更新重采样信号
  
else
%   Resample for the input rate higher than the output
%   Resampling includes an anti-aliasing filter（抗混叠滤波器）.
    y=resample(x,fy,fx);
    
%   Reduce the input signal bandwidth to 21 kHz:（限制输入信号带宽）
%   Chebychev Type 2 LP (smooth passband)
    order=7; %Filter order（滤波器阶数）
    atten=30; %Sidelobe attenuation in dB（旁瓣衰减）
    fcutx=21/fx; %Cutoff frequency as a fraction of the sampling rate（截止频率）
    [bx,ax]=cheby2(order,atten,fcutx);
    xfilt=filter(bx,ax,x);
    
%   Reduce the resampled signal bandwidth to 21 kHz
    fcuty=21/fy;
    [by,ay]=cheby2(order,atten,fcuty);
    yfilt=filter(by,ay,y); % 限制带宽为21kHz后的重采样信号
    
%   Compute the input and output RMS levels within the 21 kHz bandwidth
%   and match the output to the input
    xRMS=sqrt(mean(xfilt.^2));
    yRMS=sqrt(mean(yfilt.^2));  
    y=(xRMS/yRMS)*y; % 二者相匹配，更新限制带宽后的重采样信号
end
