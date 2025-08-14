function [xdB,xBM,ydB,ydB_,yBM,xSL,ySL,fsamp]=...
    eb_EarModel(x,xsamp,y,ysamp,HL,itype,Level1)
% Function to implement a cochlear model that includes the middle ear,
% auditory filter bank, OHC dynamic-range compression, and IHC attenuation.
% The inputs are the reference and processed signals that are to be
% compared. The reference x is at the reference intensity (e.g. 65 dB SPL
% or with NAL-R amplification) and has no other processing. The processed
% signal y is the hearing-aid output, and is assumed to have the same or
% greater group delay compared to the reference. The function outputs are
% the envelopes of the signals after OHC compression and IHC loss
% attenuation.
%
% Calling arguments:
% x        reference signal: should be adjusted to 65 dB SPL (itype=0 or 1)
%            or to 65 dB SPL plus NAL-R gain (itype=2)
% xsamp    sampling rate for the reference signal, Hz
% y        processed signal (e.g. hearing-aid output) includes HA gain
% ysamp    sampling rate for the processed signal, Hz
% HL       audiogram giving the hearing loss in dB at six audiometric
%            frequencies: [250, 500, 1000, 2000, 4000, 6000] Hz
% itype    purpose for the calculation: 
%          0=intelligibility: reference is nornal hearing and must not
%            include NAL-R EQ
%          1=quality: reference does not include NAL-R EQ
%          2=quality: reference already has NAL-R EQ applied
% Level1   level calibration: signal RMS=1 corresponds to Level1 dB SPL
%
% Returned values:
% xdB      envelope for the reference in each band
% xBM      BM motion for the reference in each band
% ydB      envelope for the processed signal in each band
% yBM      BM motion for the processed signal in each band
% xSL      compressed RMS average reference（参考信号的平均压缩均方值） in each band converted to dB SL
% ySL      compressed RMS average output（输出的平均压缩均方值） in each band converted to dB SL
% fsamp    sampling rate in Hz for the model outputs
%
% James M. Kates, 27 October 2011.
% BM motion added 30 Dec 2011.
% Revised 19 June 2012.
% Remove match of reference RMS level to processed 29 August 2012.
% IHC adaptation added 1 October 2012.
% BM envelope coverted to dB SL, 2 Oct 2012.
% Filterbank group delay corrected, 14 Dec 2012.

% Processing parameters
% OHC and IHC parameters for the hearing loss
% Auditory filter center frequencies span 80 to 8000 Hz.（听觉滤波器中心频率的跨度80~8000Hz）
nchan=32; %Use 32 auditory frequency bands 
mdelay=1; %Compensate for the gammatone group delay
cfreq=eb_CenterFreq(nchan); %Center frequencies on an ERB scale （ERB刻度计算中心频率）

% Cochlear model parameters for the processed signal
[attnOHCy,BWminy,lowkneey,CRy,attnIHCy]=eb_LossParameters(HL,cfreq);  %（处理信号的受损耳蜗的模型参数）

% The cochlear model parameters for the reference are the same as for the
% hearing loss if calculating quality, but are for normal hearing if
% calculating intelligibility.
if itype == 0  %itype=0是正常听力者，参考信号不含听损
    HLx=0*HL;
else
    HLx=HL;
end
[attnOHCx,BWminx,lowkneex,CRx,attnIHCx]=eb_LossParameters(HLx,cfreq); % （参考信号的耳蜗模型参数）

% Parameters for the control filter bank（控制滤波器组）
HLmax=100*[1 1 1 1 1 1]; % 给出最大的听力损失
shift=0.02; %Basal shift（基础移动） of 0.02 of the basilar membrane length（基底膜长度）
cfreq1=eb_CenterFreq(nchan,shift); %Center frequencies for the control（控制滤波器的中心频率）
[~,BW1,~,~,~]=eb_LossParameters(HLmax,cfreq1); %Maximum BW for the control（求出控制滤波器的最大带宽）

% ---------------------------------------
% Input signal adjustments
% Force the input signals to be row vectors
x=x(:)';
y=y(:)';



% Add NAL-R equalization if the quality reference doesn't already have it.
if itype == 1
%     nfir=140; %Length in samples（采样长度） of the FIR NAL-R EQ filter (24-kHz rate)
%     [nalr,~]=eb_NALR(HL,nfir,xsamp); %Design the NAL-R filter
%     xnal=conv(x,nalr); %Apply the NAL-R filter
%     x=xnal(nfir+1:nfir+length(x));
 
    x=Fig6_Amplification(HL,x',xsamp);
    x=x';
%     audiowrite('p232_001(20)_inner.wav',x,xsamp);
%     
%     figure(1);
%     subplot(3,1,2);
%     xlen = length(x24nal);
%     plot((1:xlen)/24000, x24nal);
%     title('线性验配公式放大后的参考信号','Fontsize',9,'Fontname','宋体')
end

% Convert the signals to 24 kHz sampling rate. Using 24 kHz guarantees that
% all of the cochlear filters have the same shape（相同的形状） independent of the
% incoming signal samplig rates（输入信号的采样率）
[x24,~]=eb_Resamp24kHz(x,xsamp);
[y24,fsamp]=eb_Resamp24kHz(y,ysamp);

% Check the file sizes
nxy=min(length(x24),length(y24));
x24=x24(1:nxy);
y24=y24(1:nxy);

% Bulk broadband signal alignment
[x24,y24]=eb_InputAlign(x24,y24);
nsamp=length(x24);

% ---------------------------------------
% Cochlear model
% Middle ear（中耳模型）
xmid=eb_MiddleEar(x24,fsamp)';
ymid=eb_MiddleEar(y24,fsamp)';

% Initialize storage（初始化存储）
% Reference and processed envelopes and BM motion
xdB=zeros(nchan,nsamp); % 每个通道中各个中心频率点
ydB=xdB;
ydB_=xdB;
xBM=xdB;
yBM=ydB;
% Reference and processed average spectral（平均频谱） values
xave=zeros(nchan,1); %Reference % 每个通道的平均值
yave=xave; %Processed
xcave=xave; %Reference control （参考控制）
ycave=yave; %Processed control
% Filter bandwidths adjusted for intensity （参考、处理信号的控制滤波器带宽，根据强度调整）
BWx=zeros(nchan,1);
BWy=BWx;

% Loop over each filter in the auditory filter bank（循环遍历听觉滤波器组中的每个滤波器）
for n=1:nchan
%   Control signal envelopes for the reference and processed signals（控制信号的包络）
    [xcontrol,ycontrol]=...
        eb_GammatoneEnv2(xmid,BW1(n),ymid,BW1(n),fsamp,cfreq1(n)); % 相同的控制滤波器的带宽
    
%   Adjust the auditory filter bandwidths for the average signal level（根据信号能量调整带宽）
    BWx(n)=eb_BWadjust(xcontrol,BWminx(n),BW1(n),Level1); %Reference
    BWy(n)=eb_BWadjust(ycontrol,BWminy(n),BW1(n),Level1); %Processed

%   Envelopes and BM motion of the reference and processed signals
    [xenv,xbm,yenv,ybm]=...
        eb_GammatoneBM2(xmid,BWx(n),ymid,BWy(n),fsamp,cfreq(n)); % 根据能量调整后的带宽
    
%   RMS levels of the ref and output envelopes for linear metric
	xave(n)=sqrt(mean(xenv.^2)); %Ave signal mag in each band（每个频带的信号平均幅度）
	yave(n)=sqrt(mean(yenv.^2));
    xcave(n)=sqrt(mean(xcontrol.^2)); %Ave control signal（每个频带控制信号的平均幅度）
    ycave(n)=sqrt(mean(ycontrol.^2));
    
%   Cochlear compression for the signal envelopes and BM motion（信号包络和基底膜运动的耳蜗压缩）
	[xc,xb]=eb_EnvCompressBM(xenv,xbm,xcontrol,attnOHCx(n),lowkneex(n),...
        CRx(n),fsamp,Level1);
	[yc,yb]=eb_EnvCompressBM(yenv,ybm,ycontrol,attnOHCy(n),lowkneey(n),...
        CRy(n),fsamp,Level1);
    yc_=yc;
    
%   Correct for the delay between the reference and output
    yc=eb_EnvAlign(xc,yc); %Align processed envelope to reference（包络的校准）
    yb=eb_EnvAlign(xb,yb); %Align processed BM motion to reference （基底膜运动的校准）

%   Convert the compressed envelopes and BM vibration envelopes to dB SL（转换成dB SL的形式）
  	[xc,xb]=eb_EnvSL2(xc,xb,attnIHCx(n),Level1);
	[yc,yb]=eb_EnvSL2(yc,yb,attnIHCy(n),Level1);
    yc_=eb_EnvSL(yc_,attnIHCy(n),Level1);
    
%   Apply the IHC rapid and short-term adaptation （IHC的快速短期适应）
    delta=2.0; %Amount of overshoot（超调量）
    [xdB(n,:),xb]=eb_IHCadapt(xc,xb,delta,fsamp);
    [ydB(n,:),yb]=eb_IHCadapt(yc,yb,delta,fsamp);
    ydB_(n,:)=eb_IHCadapt1(yc_,delta,fsamp);
    
%   Additive noise level to give the auditory threshold
    IHCthr=-10.0; %Additive noise level, dB re: auditory threshold
    xBM(n,:)=eb_BMaddnoise(xb,IHCthr,Level1);
    yBM(n,:)=eb_BMaddnoise(yb,IHCthr,Level1);
end

% Correct for the gammatone filterbank interchannel group delay.
% Function eb_EnvAlign matches the processed signal delay to the reference,
% so the filterbank delay correction should be for the
% reference.（函数eb_EnvAlign将处理后的信号与参考信号进行了匹配，因此滤波器延迟校正是针对参考信号）
if mdelay > 0
    xdB=eb_GroupDelayComp(xdB,BWx,cfreq,fsamp);
    ydB=eb_GroupDelayComp(ydB,BWx,cfreq,fsamp);
    xBM=eb_GroupDelayComp(xBM,BWx,cfreq,fsamp);
    yBM=eb_GroupDelayComp(yBM,BWx,cfreq,fsamp);
end

% Convert average gammatone outputs to dB SL
xSL=eb_aveSL(xave,xcave,attnOHCx,lowkneex,CRx,attnIHCx,Level1);
ySL=eb_aveSL(yave,ycave,attnOHCy,lowkneey,CRy,attnIHCy,Level1);
end
