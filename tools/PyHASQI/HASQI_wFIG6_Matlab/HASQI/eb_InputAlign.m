function [xp, yp]=eb_InputAlign(x,y)
% Function to provide approximate temporal alignment of the reference and
% processed output signals. Leading and trailing zeros are then pruned. 
% The function assumes that the two sequences have the same sampling rate: 
% call eb_Resamp24kHz for each sequence first, then call this function to
% align the signals.��ͬ���ο��źźʹ������źţ�
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
% purposes of computing the cross-covariance������أ�
nx=length(x);
ny=length(y);
nsamp=min(nx,ny); % ƥ���źų����Լ��㻥���

% Determine the delay of the output relative to the reference
xy=xcov(x(1:nsamp),y(1:nsamp)); %Cross-covariance of the ref and output
[~,index]=max(abs(xy)); %Find the maximum value���������������������
delay=nsamp-index; % ȷ���������ڲο��źŵ��ӳ�

% Back up 2 msec to allow for dispersion ��
fsamp=24000; %Cochlear model input sampling rate in Hz������ģ������Ĳ�����/Hz��
delay=delay - 2*fsamp/1000; %Back up 2 msec ����Ϊ��y��ǰ��

% Align the output with the reference allowing for the dispersion
if delay > 0
%   Output delayed relative to the reference ������ӳ�
    y=[y(delay+1:ny) zeros(1,delay)]; %Remove the delay���ӳ���ǰ�Ʋ����㣩
else
%   Output advanced relative to the reference �����ǰ
    delay=-delay;
    y=[zeros(1,delay) y(1:ny-delay)]; %Add advance ����ǰֵ���Ʋ����㣩
end

% Find the start and end of the noiseless reference sequence���Ҳο��źŵ���ʼ��ĩβ���޼��źţ�
xabs=abs(x);
xmax=max(xabs);
xthr=0.001*xmax; %Zero detection threshold��0�����ֵ��
for n=1:nx
%	First value above the threshold working forwards from the beginning
    if xabs(n)>xthr  % ��¼��ǰ����һ��������ֵ�ĵ�ı��
		nx0=n;
		break;
    end
end
for n=nx:-1:1
%	First value above the threshold working backwards from the end
	if xabs(n)>xthr % ��¼�Ӻ���ǰ��һ��������ֵ�ĵ�
		nx1=n;
		break;
	end
end

% Prune��ɾ�����޼��� the sequences to remove the leading and trailing zeros
if nx1>ny; nx1=ny; end;
xp=x(nx0:nx1);
yp=y(nx0:nx1);



