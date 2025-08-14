function [y,b]=eb_EnvSL2(env,bm,attnIHC,Level1)
% Function to convert the compressed envelope returned by
% cochlea_envcomp to dB SL.
%
% Calling arguments
% env			linear envelope after compression��ѹ��������԰��磩
% bm            linear basilar membrane vibration after compression ��ѹ��������Ի���Ĥ�𶯣�
% attnIHC		IHC attenuation at the input to the synapse
% Level1		level in dB SPL corresponding to 1 RMS
%
% Return
% y				envelope in dB SL
% b             BM vibration with envelope converted to dB SL ������ת��ΪdB SL�µĻ���Ĥ��ֵ��
%
% James M. Kates, 20 Feb 07.
% IHC attenuation added 9 March 2007.
% Basilar membrane vibration conversion added 2 October 2012.

% Convert the envelope to dB SL
small=1.0e-30; %To prevent taking log of 0 ��ֹ����0��ֵ�Ķ���
env1 = env+small;
env1 = env1./5e-4;
y=Level1 - attnIHC + 20*log10(env1); % ���԰���ת��ΪdB SL(�о���)
y=max(y,0.0);

% Convert the linear BM motion to have a dB SL envelope
gain=(y + small)./(env + small); %Gain that converted the env from lin to dB SPL
b=gain.*bm; %Apply gain to BM motion
end
