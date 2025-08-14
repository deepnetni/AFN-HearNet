function [k,b] = Fit_FIG6(audiogram_f, audiogram_ht, ChannelNum)
% ʹ��FIG6������ʽ�������䣬��������ͼ�õ���ͨ��WDRC IO���߲���
%
% ���룺
% audiogram_f:  ����ͼ���������Ӧ��ͨ�����
% audiogram_ht: ����ͼ������������
% ChannelNum:   ͨ������
% 
% �����
% k,b:          IO���߱��ʽSPL_out = k * SPL_in + b�е�k��b��ÿһ�ж�Ӧһ��ͨ����ÿһ�ж�Ӧһ����ѹ��

%%
%---------------  ͨ������,���ͨ���Լ�����Ƶ�㸳ֵ-------------------------
if ChannelNum==4
    %SPL_offset = 95.1002986*ones(1,ChannelNum);
    ChannelNum_fc = [500,1000,2000,4000 ];
    ChannelNum_ft=[0,750,1500,3000,8000];
elseif ChannelNum==6
    %SPL_offset = 94.9133059*ones(1,ChannelNum);
    ChannelNum_fc = [ 250,500,1000,2000,3000,4000 ];
    ChannelNum_ft=[0,250,625,1375,2500,3500,8000];
elseif ChannelNum==8
    %SPL_offset = 95.1079128*ones(1,ChannelNum);
    ChannelNum_fc = [ 250,500,750,1125,1750,2500,4000,6000 ];
    ChannelNum_ft=[0,250,500,750,1375,2500,3500,4875,8000];
elseif ChannelNum==12
    %SPL_offset = 95.31420495*ones(1,ChannelNum);
    ChannelNum_fc = [ 250,375,500,750,1000,1375,1750,2250,3000,3875,4875,6250 ];
    ChannelNum_ft=[0,250,375,500,750,1125,1500,1875,2625,3375,4250,5625,8000];
elseif ChannelNum==16
    %SPL_offset = 96.7119344*ones(1,ChannelNum);
    ChannelNum_fc = [ 250,375,500,625,750,1000,1125,1375,1750,2125,2625,3125,3875,4625,5500,6625 ];
    ChannelNum_ft=[0,250, 375,500,625,750,1000,1250,1625,2000,2375,2875,3500,4250,5125,6125,8000];
end
%%
%---------------------------- ����ͨ�����ֺ󣬸�ͨ����Ӧ�����������ɵ���������HT ---------------
HT=zeros(1,ChannelNum);
for j = 1:ChannelNum
	HT(j) = CalculateHL_LinearFitting(ChannelNum_fc(j), audiogram_f, audiogram_ht,length(audiogram_ht));
end 

% ��ÿ��ͨ�����㲻ͬ��ѹ����k��b��
k = zeros(ChannelNum,3);                                                    % ����ֱ��б��
b = zeros(ChannelNum,3);                                                    % ����ֱ�߽ؾ�
%ά�� ChannelNum*3  ��������

tklin = 40;                                                                 % ��Ƶ����Ƶ��
tkhin = 60;                                                                 % ��Ƶ����Ƶ��

%% ----------------  �ò����Ѽ����� ig_Update������  �ýű��ɲ�ʹ��
%{
Limit_1=2000;
Limit_2=3000;

GainLimith=28;
GainLimitm=24;
GainLimitl=20;

GainLimitHT_l=4;
GainLimitHT_m=6;
GainLimitHT_h=8;
%}
%% ----------------- ͨ�����ֺ󣬸�������ͼ��ȡHT����,����FIG6��ʽ���䣬��ȡÿ��I/Oͼ������б�ʺͽؾ�
for i = 1:ChannelNum
    ht = HT(i);
	k(i,1) = 1;
    
    if ht < 20
        b(i,1) = 0;
        k(i,2) = 1;
        b(i,2) = 0;
        k(i,3) = 1;
        b(i,3) = 0;
        
    elseif ht < 40
        
        %------------------��һ��---------------------
        ig40 = ht - 20;
        %---------------�����Ƿ����-------------
        %ig40=ig_Update(40,ig40,ChannelNum_ft(i));
        
        splout40 = 40 + ig40;                                               % splout = splin + ig
        b(i,1) = splout40 - 40;
        tklout = tklin + b(i,1);         %thlin 40  thhin 60

        %------------------�ڶ���---------------------
        ig65 = 0.6 * (ht - 20);
        %----------�����Ƿ����-------------        
        %ig65=ig_Update(65,ig65,ChannelNum_ft(i));
        
        splout65 = 65 + ig65;
        k(i,2) = (splout65 - tklout) / (65 - tklin);
        b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin);
        tkhout = k(i,2) * tkhin + b(i,2);

        %------------------������---------------------
        ig95 = 0;
        %----------�����Ƿ����-------------    
        %ig95=ig_Update(95,ig95,ChannelNum_ft(i));
        
        splout95 = 95 + ig95;
        k(i,3) = (splout95 - tkhout) / (95 - tkhin);
        b(i,3) = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin);
        
    elseif ht < 60       
        %------------------��һ��---------------------
        ig40 = ht - 20;
        %---------------�����Ƿ����-------------
        %ig40=ig_Update(40,ig40,ChannelNum_ft(i));
        
        splout40 = 40 + ig40;
        b(i,1) = splout40 - 40;
        tklout = tklin + b(i,1);
        
        %------------------�ڶ���---------------------
        ig65 = 0.6 * (ht - 20);
        %----------�����Ƿ����-------------        
        %ig65=ig_Update(65,ig65,ChannelNum_ft(i));
        
        splout65 = 65 + ig65;
        k(i,2) = (splout65 - tklout) / (65 - tklin);
        b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin);
        tkhout = k(i,2) * tkhin + b(i,2);
        
        %------------------������---------------------
        ig95 = 0.1*(ht - 40)^1.4;
        %----------�����Ƿ����-------------    
        %ig95=ig_Update(95,ig95,ChannelNum_ft(i));
        
        splout95 = 95 + ig95;
        k(i,3) = (splout95 - tkhout) / (95 - tkhin);
        b(i,3) = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin);
    else
        %------------------��һ��---------------------
        ig40 = ht - 20 - 0.5 * (ht - 60);
        %---------------�����Ƿ����-------------
        %ig40=ig_Update(40,ig40,ChannelNum_ft(i));
  
        splout40 = 40 + ig40;
        b(i,1) = splout40 - 40;
        tklout = tklin + b(i,1);

        %------------------�ڶ���---------------------
        ig65 = 0.8 * ht - 23;
        %----------�����Ƿ����-------------        
        %ig65=ig_Update(65,ig65,ChannelNum_ft(i));

        splout65 = 65 + ig65;
        k(i,2) = (splout65 - tklout) / (65 - tklin);
        b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin);
        tkhout = k(i,2) * tkhin + b(i,2);

        %------------------������---------------------
        ig95 = 0.1*(ht - 40)^1.4;
        %----------�����Ƿ����-------------    
        %ig95=ig_Update(95,ig95,ChannelNum_ft(i));
     
        splout95 = 95 + ig95;
        k(i,3) = (splout95 - tkhout) / (95 - tkhin);
        b(i,3) = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin);
    end
end