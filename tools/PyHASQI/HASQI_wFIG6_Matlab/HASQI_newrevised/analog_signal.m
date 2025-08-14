clear
close all
clc

fs=2000;                                                                   %采样频率
N=2000;                                                                    %采样点数
t1=1:N;                                                                    %时间长度
f=fs*(0:N-1)/N;                                                            %频率分布
t=t1/fs;                                                                   %时间（s�?
max_t1=max(t);
Al=1;                                                                      %有效信号幅�?
A_noise=0.6;
f0=300;                                                                    %共振频率
zeta0=0.1;                                                                 %阻尼系数，轴承故障信号阻尼系数较大，衰减较快
t0=0.05;                                                                   %延时
fi=20;                                                                     %故障频率
T=1/fi;
dt=1/fs;
Ws=fs*T;                                                                   %Laplace小波支撑长度(以点数表�?

sig1=exp(-(zeta0/sqrt(1-zeta0^2))*2*pi* f0*(t1/fs)).*sin(2*pi* f0*(t1/fs));
Wss=round(t0*fs);
for k=1:N
if k<=Wss
    sig2(k)=0;
else
    sig2(k)=sig1(k-Ws);
end
end
signal=sig2;
for i=1:N/(T*fs)
    Wss=round(T*fs*i+t0*fs);
    for k=1:N
        if k>Wss
            signal(k)=signal(k)+sig1(k-Wss);
        end
    end
end
noise=A_noise*randn(1,N);                                                  %噪声信号
y_original=signal+noise;                                                   %含噪信号

figure(1) %�������ź�
plot(t,signal,'k')
set(gca,'fontname','Times New Roman')
xlabel('t(s)','fontname','Times New Roman')
ylabel('�ź�','fontname','Times New Roman')
set(gcf,'Position',[200,200,1000,200]);
xlim([0,max(t)])
ylim([min(signal)-0.2,max(signal)+0.2])

figure(2) %�����ź�
plot(t,y_original,'k')
set(gca,'fontname','Times New Roman')
xlabel('t(s)','fontname','Times New Roman')
ylabel('�ź�','fontname','Times New Roman')
set(gcf,'Position',[200,200,1000,200]);
xlim([0,max(t)])
ylim([min(y_original)-0.2,max(y_original)+0.2])
%---------------------------计算信噪�?
f_hat =abs(fft(y_original));
ps=sum((signal).^2)/length(signal);
pn=sum((noise).^2 )/length(noise);
SNR=10*log10(ps/pn);
