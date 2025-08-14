clear all;
addpath('.\HASPI_HASQI');   
addpath('.\WDRC_0903');
% addpath('F:\HASQI_JMJ\python');

rand('seed',666);   % 将生成的随机数暂时保存
mode='test';    % 设定模式
% train_data_path='.\test_data_eng_level';
% if ~exist(train_data_path, 'dir')
%     mkdir(train_data_path);
% end
% % % 失真语音的路径
% distorted_path={'D:\HASQI\dataset\train\mandarin\noisy', ...       
%                 'D:\HASQI\dataset\train\mandarin\wiener_PS', ...
%                 'D:\HASQI\dataset\train\mandarin\wiener_DD', ...
%                 'D:\HASQI\dataset\train\mandarin\NS'};
% distorted_path={'.\distorted_eng'};    % 经过加噪后降噪处理的语音
% % 参考语音的路径          
% clean_path='.\clean_eng';
% % distorted_path={'.\testset\test\distorted'};
% % clean_path='.\testset\test\clean';
% fileFolder=fullfile(distorted_path{1}); % 返回包含文件完整路径的字符向量
% dirOutput=dir(fullfile(fileFolder,'*wav'));   %列出目录下的wav文件
% fileNames={dirOutput.name}; %处理后失真语音的文件名

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

% 读取听障患者、听力补偿信息
m=xlsread('NAL_0411.xlsx'); 
m=m(2:end,4:end);
% if strcmp(mode,'train')
%     m=m(1:100,:);   %读取听力图数据
% else 
%     m=m(101:end,:);
% end
ht_data=m(:,1:6);   % 将其存在列表中，每一行表示一个患耳
num_ht=size(ht_data,1); % 计算患耳数量
gains=m(:,7:end);   % 读取对应的NL2增益数据（每一行表示一个患耳）
gains=reshape(gains,[size(gains,1),19,7]); % 重构增益，转换为三维数组（每一个矩阵表示一个声压级，矩阵中的每一行表示一个患耳的增益） 
gains=permute(gains,[1,3,2]); % 置换数组维度，行数不变，列数和个数置换，（共有19个矩阵，每个矩阵表示一个通道下的各声压级患耳的增益）

config_NAL;
% 
Level1 = 65;
eq = 2;

% ht_data=xlsread('.\HT.xlsx');
% ht_data=ht_data(2:end,:);
% num_ht=size(ht_data,1);
% ht0=[12,5,3,-2,-5,4];

% f1=fullfile(train_data_path,'features.bin');
% fid1=fopen(f1,'wb');  % 写入数据
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
%     fprintf('Processing %s\n',distorted_path{i}); % 写出正在处理的失真信号路径
%     for j=1:1  % 分别处理每个失真信号文件
%         idx=strfind(fileNames{j},'_');
        dist_file='./noise.wav';
%         clean_name=fileNames{j}(1:idx(2)-1);    % 找到对应的纯净语音文件
%         clean_name=fileNames{j};    % distorted文件夹内音频名称对应
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
            while s<=0  % 初始化设定
%                 FBEs=[];
%                 scores=[];
%                 vads=[];
%                 nf=[];
%                 levels=[];
%                 hts=[];
%                 p=ceil(rand()*(num_ht/2));  % ceil向上舍入
%                 for a=1:2
                    idx=5;
                    g=squeeze(gainsTN(idx,:,:)); % 处理好的每一患耳对应的各声压级下各通道的增益
                    ht=ht_data(idx,:);  % 对应患耳的听力图数据
                    average_ht=(ht(2)+ht(3)+ht(4)+ht(5))/4; % 听损等级
                    if average_ht < 26
                        level = 0;    % 正常
                    elseif average_ht < 41
                        level = 1;    % 轻度聋
                    elseif average_ht < 61
                        level = 2;    % 中度聋
                    elseif average_ht < 81
                        level = 3;    % 重度聋
                    else
                        level = 4;    % 极重度聋
                    end
%                     fprintf('j:%d\n',j);
                    fprintf('idx: %d  level: %d\n',idx,level);
%                     
                    if rand()>1;    % 这可能吗？
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
%                     scores=[scores,score]; % 填充综合评分矩阵
%                     FBEs=[FBEs,FBE];
% %                     vads=[vads;vad];
%                     levels=[levels;level];
%                     nf=[nf,size(FBE,2)];  % FBE的列向量数（每段时域语音的点数）
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
%                     if mod(j,100)==0    % 每处理100个文件，就输出一段文件处理结束标语
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
    
    
