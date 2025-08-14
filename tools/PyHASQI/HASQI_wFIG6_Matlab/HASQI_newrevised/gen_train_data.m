clear all;
addpath('.\HASPI_HASQI');   
addpath('.\WDRC_0903');
% addpath('F:\HASQI_JMJ\python');

rand('seed',666);   % �����ɵ��������ʱ����
mode='test';    % �趨ģʽ
% train_data_path='.\test_data_eng_level';
% if ~exist(train_data_path, 'dir')
%     mkdir(train_data_path);
% end
% % % ʧ��������·��
% distorted_path={'D:\HASQI\dataset\train\mandarin\noisy', ...       
%                 'D:\HASQI\dataset\train\mandarin\wiener_PS', ...
%                 'D:\HASQI\dataset\train\mandarin\wiener_DD', ...
%                 'D:\HASQI\dataset\train\mandarin\NS'};
% distorted_path={'.\distorted_eng'};    % ����������봦�������
% % �ο�������·��          
% clean_path='.\clean_eng';
% % distorted_path={'.\testset\test\distorted'};
% % clean_path='.\testset\test\clean';
% fileFolder=fullfile(distorted_path{1}); % ���ذ����ļ�����·�����ַ�����
% dirOutput=dir(fullfile(fileFolder,'*wav'));   %�г�Ŀ¼�µ�wav�ļ�
% fileNames={dirOutput.name}; %�����ʧ���������ļ���

% m=csvread('.\result.csv');
% num_ht=(size(m,1)-17)/19+1;
% amf=m(1,1:11);
% gains=zeros(num_ht*14,size(m,2));
% for i=1:num_ht
%     gains((i-1)*14+1:i*14,:)=m((i-1)*19+4:i*19-2,:);
% end
% gains=gains(:,3:end);
% ht_data=m(2:19:end,:);
% ht_data=ht_data(:,1:11);
% for i=1:num_ht
%     idx=find(ht_data(i,:)>0);
%     f=amf(idx);
%     ht=ht_data(i,idx);
%     if f(1)>amf(1)
%         f=[amf(1),f];
%         ht=[ht(1),ht];
%     end
%     if f(end)<amf(end)
%         f=[f,amf(end)];
%         ht=[ht,ht(end)];
%     end
%     ht_data(i,:)=interp1(f,ht,amf);
% end
% 
% gains=reshape(gains',[size(gains,2),14,size(gains,1)/14]);
% gains=permute(gains,[3,2,1]);
% % gains=gains(:,4:end,:);
% gains=gains(:,2:2:end,:);

% ��ȡ���ϻ��ߡ�����������Ϣ
m=xlsread('NAL_0411.xlsx'); 
m=m(2:end,4:end);
% if strcmp(mode,'train')
%     m=m(1:100,:);   %��ȡ����ͼ����
% else 
%     m=m(101:end,:);
% end
ht_data=m(:,1:6);   % ��������б��У�ÿһ�б�ʾһ������
num_ht=size(ht_data,1); % ���㻼������
gains=m(:,7:end);   % ��ȡ��Ӧ��NL2�������ݣ�ÿһ�б�ʾһ��������
gains=reshape(gains,[size(gains,1),19,7]); % �ع����棬ת��Ϊ��ά���飨ÿһ�������ʾһ����ѹ���������е�ÿһ�б�ʾһ�����������棩 
gains=permute(gains,[1,3,2]); % �û�����ά�ȣ��������䣬�����͸����û���������19������ÿ�������ʾһ��ͨ���µĸ���ѹ�����������棩

config_NAL;
% 
Level1 = 65;
eq = 2;

% ht_data=xlsread('.\HT.xlsx');
% ht_data=ht_data(2:end,:);
% num_ht=size(ht_data,1);
% ht0=[12,5,3,-2,-5,4];

% f1=fullfile(train_data_path,'features.bin');
% fid1=fopen(f1,'wb');  % д������
% fid2=fopen(fullfile(train_data_path,'HASQI.bin'),'wb');
% fid3=fopen('.\test_data_eng_level\num_frames.bin','wb');
% % fid4=fopen([train_data_path,'\','VAD.bin'],'wb');
% fid5=fopen([train_data_path,'\','level.bin'],'wb');
% fid6=fopen([train_data_path,'\','ht.bin'],'wb');
% 
% labels={};
% cnt=0;
num_path = 1;
% for i=1: num_path
%     fprintf('Processing %s\n',distorted_path{i}); % д�����ڴ����ʧ���ź�·��
%     for j=1:1  % �ֱ���ÿ��ʧ���ź��ļ�
%         idx=strfind(fileNames{j},'_');
        dist_file='./noise.wav';
%         clean_name=fileNames{j}(1:idx(2)-1);    % �ҵ���Ӧ�Ĵ��������ļ�
%         clean_name=fileNames{j};    % distorted�ļ�������Ƶ���ƶ�Ӧ
        clean_file='./clean.wav';
        
%         figure(1);
%         subplot(211);
%         [s1,~]=audioread(clean_file);
%         cn=length(s1);
%         plot((1:cn)/SampleRate, s1);
%         figure(2);
%         subplot(211);
%         [s2,~]=audioread(dist_file);
%         dn = length(s2);
%         plot((1:dn)/SampleRate, s2);
        
%         for t=1:1
            s=0;
            while s<=0  % ��ʼ���趨
%                 FBEs=[];
%                 scores=[];
%                 vads=[];
%                 nf=[];
%                 levels=[];
%                 hts=[];
%                 p=ceil(rand()*(num_ht/2));  % ceil��������
%                 for a=1:2
                    idx=5;
                    g=squeeze(gainsTN(idx,:,:)); % ����õ�ÿһ������Ӧ�ĸ���ѹ���¸�ͨ��������
                    ht=ht_data(idx,:);  % ��Ӧ����������ͼ����
                    average_ht=(ht(2)+ht(3)+ht(4)+ht(5))/4; % ����ȼ�
                    if average_ht < 26
                        level = 0;    % ����
                    elseif average_ht < 41
                        level = 1;    % �����
                    elseif average_ht < 61
                        level = 2;    % �ж���
                    elseif average_ht < 81
                        level = 3;    % �ض���
                    else
                        level = 4;    % ���ض���
                    end
%                     fprintf('j:%d\n',j);
                    fprintf('idx: %d  level: %d\n',idx,level);
%                     
                    if rand()>1;    % �������
                        g=0*g;
                        ht=0*ht; 
                    end
            %         ht=[0    50    50     0    55     0    55    55    55];
        %             ht=zeros(1,9);
        %             [x,fx]=WDRC(clean_file,[ht(1),ht(2),ht(3),ht(5),ht(7),ht(9)]);
        %             [y,fy]=WDRC(dist_file,[ht(1),ht(2),ht(3),ht(5),ht(7),ht(9)]);       
        %             hl=[ht(1),ht(2),ht(3),ht(5),ht(7),ht(8)]-20;
            %         hl=[ht(1),ht(2),ht(3),ht(5),ht(7),ht(8)]-ht0;
                    [x,fx]=WDRC_NAL(clean_file,g);
                    [y,fy]=WDRC_NAL(dist_file,g);     
        %             hl=[ht(2),ht(3),ht(5),ht(7),ht(9),ht(10)];
        
%                     figure(1);
%                     subplot(212);
%                     xn = length(x);
%                     plot((1:xn)/SampleRate, x);
%                     figure(2);
%                     subplot(212);
%                     yn = length(y);
%                     plot((1:yn)/SampleRate, y);
                    
                    hl=ht;
                    hl=max(0,hl);
                    [score,~,~]=HASQI(x,fx,y,fy,hl,Level1,eq,mode);
%                     scores=[scores,score]; % ����ۺ����־���
%                     FBEs=[FBEs,FBE];
% %                     vads=[vads;vad];
%                     levels=[levels;level];
%                     nf=[nf,size(FBE,2)];  % FBE������������ÿ��ʱ�������ĵ�����
%                 end
%                 s=min(scores);
                    fprintf('score: %.4f \n',score);
%                 if s>0
%                     cnt=cnt+1;
%                     labels{cnt}=[clean_name,'_',num2str(p,'%02d'),'l'];
%                     cnt=cnt+1;
%                     labels{cnt}=[clean_name,'_',num2str(p,'%02d'),'r'];
%                     scores=[scores,score];
%                     FBEs=[FBEs,FBE];
%                     levels=[levels,level];
%                     nf=[nf,size(FBE,2)];
%                     hts=[hts,hl];
%                     FBEs=FBEs-384;
%                     
%                     fwrite(fid1,FBEs,'float'); 
%                     fwrite(fid2,scores,'float');
%                     fwrite(fid3,nf,'int32');
% %                     fwrite(fid4,vads,'float');
%                     fwrite(fid5,levels,'int32');
%                     fwrite(fid6,hts,'int32');
%                     
%                     if mod(j,100)==0    % ÿ����100���ļ��������һ���ļ������������
%                        fprintf('%d/%d completed\n',j,length(fileNames));
%                     end
%                 end
            end
%         end
%     end
% end
% fclose(fid1);
% fclose(fid2);
% fclose(fid3);
% % fclose(fid4);
% fclose(fid5);
% fclose(fid6);
    
    
