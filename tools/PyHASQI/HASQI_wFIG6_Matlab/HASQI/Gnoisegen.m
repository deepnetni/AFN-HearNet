function [y,noise] = Gnoisegen(x,snr)
 
noise=randn(size(x));              % 用randn函数产生高斯白噪声
 
Nx=length(x);                      % 求出信号x长
 
signal_power = 1/Nx*sum(x.*x);     % 求出信号的平均能量
 
noise_power=1/Nx*sum(noise.*noise);% 求出噪声的能量
 
noise_variance = signal_power / ( 10^(snr/10) );    % 计算出噪声设定的方差值
 
noise=sqrt(noise_variance/noise_power)*noise;       % 按噪声的平均能量构成相应的白噪声
 
y=x+noise;   % 合成带噪语音
end
