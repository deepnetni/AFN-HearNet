clc;
clear all;

% ----------------------������������---------------------- %
[x,fx] = audioread('fileid_132_clean.wav');            % ��WAV�ļ�ת���ɱ���
[y,fy] = audioread('fileid_132_noisy.wav');            % ��WAV�ļ�ת���ɱ���
% ---------------------����������в���--------------------- %

HL = [80, 85, 90, 80, 90, 80];
% HL = [50, 50, 50, 50, 50, 50];
% HL = [40, 40, 40, 40, 40, 40];
% HL = [20, 40, 30, 35, 40, 50];
Level1 = 65;
eq = 1;
% mode='train';
x_fig6 = Fig6_Amplification(HL,x,fx);
y_fig6 = Fig6_Amplification(HL,y,fy);
% [Intel,raw1] = HASPI_v1(x,fx,y,fy,HL,Level1)
% [Combined,Nonlin,Linear,raw] = HASQI_v2(x,fx,y,fy,HL,eq,Level1);
% [hasqi,FBE,vad]=HASQI_fig6(x,fx,x_fig6,fx,HL,Level1,eq,mode);
[Combined,Nonlin,Linear,raw] = HASQI_v2(x,fx,y_fig6,fy,HL,eq,Level1);
 
 
 
