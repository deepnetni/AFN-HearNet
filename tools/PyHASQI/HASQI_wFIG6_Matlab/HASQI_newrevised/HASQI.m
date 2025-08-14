function [hasqi,FBE,vad]=HASQI(x,fx,y,fy,HL,Level1,eq,mode)
% [x,fx] = audioread(ref);              % ��WAV�ļ�ת���ɱ���
% xm = max(abs(x));         % �ҳ�˫������ֵ
% x = x/xm;                                       % ��һ������
r = rms(x);
x = x / r;
x = x';                                         % ��x��������תΪ������

% [y,fy] = audioread(dis);                % ��WAV�ļ�ת���ɱ���
% ym = max(abs(y));         % �ҳ�˫������ֵ
% y = y/ym;                                       % ��һ������
% y = y / r;                  % ������ѵ������ʱ���òο�������rms��һ��ʧ�������������ɲ�������ʱ����ʧ��������rms��һ��ʧ������
if strcmp(mode, 'test')
    y1 = y / rms(y);
    y1 = y1';                                         % ��x��������תΪ������
    [~,~,~,~,FBE,vad] = HASQI_v2(x,fx,y1,fy,HL,eq,Level1);

    y2 = y / r;
    y2 = y2';
    [hasqi,~,~,~,~,~] = HASQI_v2(x,fx,y2,fy,HL,eq,Level1);
elseif strcmp(mode, 'train')
    y = y / r;
    y = y';
    [hasqi,~,~,~,FBE,vad] = HASQI_v2(x,fx,y,fy,HL,eq,Level1);
end