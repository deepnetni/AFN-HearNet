function [y,b]=eb_EnvCompressBM(envsig,bm,control,attnOHC,thrLow,CR,fsamp,Level1)
% Function to compute the cochlear compression in one auditory filter
% band.��ʵ�������˲����Ķ���ѹ����
% The gain is linear below the lower threshold, compressive with
% a compression ratio of CR:1 between the lower and upper thresholds,
% and reverts to linear above the upper threshold. The compressor
% assumes that auditory threshold is 0 dB SPL.������-ѹ��-���ԣ�
%
% Calling variables:
% envsig	analytic signal envelope (magnitude) �������źŰ��磩returned by the 
%			gammatone filter bank
% bm        BM motion output by the filter bank ���˲���������Ļ���Ĥ�˶���
% control	analytic control envelope���������ư��磩 returned by the wide control
%			path filter bank
% attnOHC	OHC attenuation at the input to the compressor��ѹ��������˵�OHC˥����
% thrLow	kneepoint for the low-level linear amplification ���ͽ����ԷŴ�Ĺյ㣩
% CR		compression ratio
% fsamp		sampling rate in Hz
% Level1	dB reference level: a signal having an RMS value of 1 is
%			assigned to Level1 dB SPL.
%
% Function outputs:
% y			compressed version of the signal envelope ��ѹ������źŰ��磩
% b         compressed version of the BM motion ��ѹ����Ļ���Ĥ�˶���
%
% James M. Kates, 19 January 2007.
% LP filter added 15 Feb 2007 (Ref: Zhang et al., 2001)
% Version to compress the envelope, 20 Feb 2007.
% Change in the OHC I/O function, 9 March 2007.
% Two-tone suppression added 22 August 2008.

% Initialize the compression parameters����ʼ��ѹ��������
thrHigh=100.0; %Upper compression threshold ������ֵ��

% Convert the control envelope to dB SPL�������������ת��dB SPL����ʽ��
small=1.0e-30;
logenv=max(control,small); %Don't want to take logarithm of zero or neg����ֹ������Ķ�����
logenv = logenv./5e-4;
logenv=Level1 + 20*log10(logenv);   % ֱ����ȥ������������úͲο��Ƚ�һ����
% ɸѡ���ڸߵ�����֮��Ķ�������
logenv=min(logenv,thrHigh); %Clip signal levels above the upper threshold
logenv=max(logenv,thrLow); %Clip signal at the lower threshold

% Compute the compression gain in dB �����������
gain=-attnOHC - (logenv - thrLow)*(1 - (1/CR));

% Convert the gain to linear and apply a LP filter to give a 0.2 msec delay
gain=10.^(gain/20); % ������ת��Ϊ����
flp=800;
[b,a]=butter(1,flp/(0.5*fsamp));  % ʹ�ý�ֹƵ��Ϊ800Hz�ĵ�ͨ������˹�˲���
gain=filter(b,a,gain); % �˲������0.2ms��ʱ��

% Apply the gain to the signals
y=gain.*envsig; %Linear envelope
b=gain.*bm; %BM motion
