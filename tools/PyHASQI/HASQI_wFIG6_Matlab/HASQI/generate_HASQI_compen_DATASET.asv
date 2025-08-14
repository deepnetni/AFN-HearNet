clear all;
rand('seed',666);   % 将生成的随机数暂时保存

train_data_path={'D:\JMCheng\DATASET\SPL_response\HASQI_TEST\5dB\umoderate',...
                 'D:\JMCheng\DATASET\SPL_response\HASQI_TEST\5dB\clean',...
                 'D:\JMCheng\DATASET\SPL_response\HASQI_TEST\5dB\umoderate\clean_fig6',...
                 'D:\JMCheng\DATASET\SPL_response\HASQI_TEST\5dB\noisy',...
                 'D:\JMCheng\DATASET\SPL_response\HASQI_TEST\5dB\umoderate\noisy_fig6'};
for i=1:length(train_data_path)
    if ~exist(train_data_path{i}, 'dir')
        mkdir(train_data_path{i});
    end
end

clean_path='D:\JMCheng\DATASET\SPL_response\HASQI_TEST\5dB\clean';
noisy_dir_path={'D:\JMCheng\DATASET\SPL_response\HASQI_TEST\5dB\noisy'};
fileFolder=fullfile(noisy_dir_path{1}); % 返回包含文件完整路径的字符向量
dirOutput=dir(fullfile(fileFolder,'*wav'));   %列出目录下的wav文件
fileNames={dirOutput.name}; %处理后失真语音的文件名


% 读取听障患者、听力补偿信息
m=xlsread('NAL_0411.xlsx'); 
m=m(2:end,4:end);

ht_data=m(:,1:6);   % 将其存在列表中，每一行表示一个患耳
num_ht=size(ht_data,1); % 计算患耳数量

Level1 = 65;
eq = 1;

fid2=fopen([train_data_path{1},'\','HASQI.txt'],'a');
fid6=fopen([train_data_path{1},'\','ht.txt'],'a');
fid7=fopen([train_data_path{1},'\','record.txt'],'a');

num_path = length(noisy_dir_path);
for i=1:num_path
    fprintf('Processing %s\n',noisy_dir_path{i}); % 写出正在处理的失真信号路径
    num_files = length(fileNames);

    for j=1:num_files  % 分别处理每个失真信号文件
        noisy_file=[noisy_dir_path{i},'\',fileNames{j}];
        name_idx=strfind(fileNames{j},'_');
%         clean_name=fileNames{j}(1:name_idx(2)-1);    % 找到对应的纯净语音文件（中文测试集）（中/英文听损）
        clean_name=fileNames{j};    % distorted文件夹内音频名称对应(中/英文训练集/英文测试集)
%         clean_file=[clean_path,'\',clean_name,'.wav'];
        clean_file=[clean_path,'\',clean_name];
        
        s=0;
        while s<=0  % 初始化设定
            idx=ceil(rand()*num_ht);  % ceil向上舍入                    
%                 g=squeeze(gainsTN(idx,:,:)); % 处理好的每一患耳对应的各声压级下各通道的增益
%             ht=ht_data(idx,:);  % 对应患耳的听力图数据

%             if rand()>1
%                 g=0*g;
%                 ht=0*ht; 
%             end

            [x, fx] = audioread(clean_file);
            [y, fy] = audioread(noisy_file);
            
%             hl=ht;  
%             hl=max(0,hl);
            hl = [40,40,40,45,40,30];
            y_fig6=Fig6_Amplification(hl,y,fy);
            x_fig6=Fig6_Amplification(hl,x,fx);
            

            new_x = zeros(1,16000 * 6);
            new_x_compen = zeros(1,16000 * 6);
            new_y_compen = zeros(1,16000 * 6);
            new_y = zeros(1,16000 * 6);
            
            if length(y_fig6)> 16000 * 6
                new_x(1:16000 * 6) = x(1:16000 * 6);
                new_x_compen(1:16000 * 6) = x_fig6(1:16000 * 6);            
                new_y(1:16000 * 6) = y(1:16000 * 6);
                new_y_compen(1:1:16000 * 6) = y_fig6(1:16000 * 6);
            else
                new_x(1:length(y_fig6)) = x(1:length(y_fig6));
                new_x_compen(1:length(y_fig6)) = x_fig6(1:length(y_fig6));
                new_y(1:length(y_fig6)) = y(1:length(y_fig6));
                new_y_compen(1:length(y_fig6)) = y_fig6(1:length(y_fig6));
            end


            [score,~,~,~]=HASQI_v2(new_x,fx,new_y_compen,fy,hl,eq,Level1);
            s=score;
            if s>0
                audiowrite([train_data_path{2},'\',sprintf('%s',clean_name)],new_x,fx);
                audiowrite([train_data_path{3},'\',sprintf('%s',clean_name)],new_x_compen,fx);
                audiowrite([train_data_path{4},'\',sprintf('%s',clean_name)],new_y,fy);                
                audiowrite([train_data_path{5},'\',sprintf('%s',clean_name)],new_y_compen,fy);
%                     fwrite(fid2,score,'float');
%                     fwrite(fid6,ht,'int32');

                fprintf(fid2,'%s,',fileNames{j}(1:end-4));
                fprintf(fid2,'%f\r\n',score);
                fprintf(fid6,'%s,',fileNames{j}(1:end-4));
                for index=1:length(hl)
                     if index==length(hl)
                         fprintf(fid6,'%d\r\n',hl(index));
                     else
                         fprintf(fid6,'%d,',hl(index));
                     end
                end

                fprintf(fid7,'%d\r\n',j);

            end
        end

    end
end

fclose(fid2);
fclose(fid6);
fclose(fid7);

