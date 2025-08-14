function [hasqi,FBE,vad]=HASQI_fig6(x,fx,y,fy,HL,Level1,eq,mode)
% [x,fx] = audioread(ref);              % 将WAV文件转换成变量
% xm = max(abs(x));         % 找出双声道极值
% x = x/xm;                                       % 归一化处理
r = rms(x);
x = x / r;
x = x';                                         % 将x从列向量转为行向量

% [y,fy] = audioread(dis);                % 将WAV文件转换成变量
% ym = max(abs(y));         % 找出双声道极值
% y = y/ym;                                       % 归一化处理
% y = y / r;                  % 当生成训练数据时，用参考语音的rms归一化失真语音；当生成测试数据时，用失真语音的rms归一化失真语音
if strcmp(mode, 'test')
    y1 = y / rms(y);
    y1 = y1';                                         % 将x从列向量转为行向量
    [~,~,~,~,FBE,vad] = HASQI_v2(x,fx,y1,fy,HL,eq,Level1);

    y2 = y / r;
    y2 = y2';
    [hasqi,~,~,~,~,~] = HASQI_v2(x,fx,y2,fy,HL,eq,Level1);
elseif strcmp(mode, 'train')
    y = y / r;
    y = y';
    [hasqi,~,~,~,FBE,vad] = HASQI_v2(x,fx,y,fy,HL,eq,Level1);
end