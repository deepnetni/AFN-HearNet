clc;
clear all;
[x,fx] = audioread('A2_0_ns.wav');              % 将WAV文件转换成变量
x=x(:)';
% x24 = x';  
% [y,fy] = audioread('p232_001(20)_inner.wav');
% 
% HL = [20, 40, 30, 35, 40, 50];
% HL = [80, 85, 90, 80, 90, 80];
% HL = [50, 50, 50, 50, 50, 50];
HL = [40, 40, 40, 40, 40, 40];
% fsamp=16000;

%  nfir=140; %Length in samples（采样长度） of the FIR NAL-R EQ filter (24-kHz rate)
%  [nalr,~]=eb_NALR(HL,nfir,fx); %Design the NAL-R filter
%  x24nal=conv(x24,nalr); %Apply the NAL-R filter
%  x24=x24nal(nfir+1:nfir+length(x24));
% audiowrite('p232_001(n40).wav',x24nal,fx);
out=Fig6_Amplification(HL,x',fx); 
% plot((1:length(out))/fx, out);
audiowrite('A2_0_dist.wav',out',fx);
disp('success!');
% out=out';

% figure(1);
% % % set(gca,'FontSize',9,'FontName','Times New Roman');
% subplot(2,1,1);
% xlen = length(x);
% plot((1:xlen)/fx, x);
% % % axis([0,1.8,-0.5,0.5]);
% title('20dB HL')
% subplot(2,1,2);
% ylen = length(y);
% plot((1:ylen)/fy, y);
% % % axis([0,1.8,-0.5,0.5]);
% title('20 inner dB HL')
% % 
% figure(2);
% plot((1:xlen)/fx, x-y);
% % % axis([0,1.8,-0.5,0.5]);
% title('差值图');
% % title('线性验配公式放大后的参考信号','Fontsize',9,'Fontname','宋体')
% % subplot(3,1,3);
% % lenout=length(out);
% % plot((1:lenout)/24000, out); 
% % title('非线性验配公式放大后的参考信号','Fontsize',9,'Fontname','宋体');
