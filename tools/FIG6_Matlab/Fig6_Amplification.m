function out=Fig6_Amplification(HL,s,fs)

SampleRate = fs;                                                         
FrameLen = 64;                                                              
FFTLen = 128;                                                               
fstep = SampleRate/FFTLen;                                                  
% ChannelNum = FFTLen/2;      
ChannelNum = 16; 
if ChannelNum==4
    SPL_offset = 95.1002986*ones(1,ChannelNum);
    ChannelNum_ft=[0,750,1500,3000,8000];
elseif ChannelNum==6
    SPL_offset = 94.9133059*ones(1,ChannelNum);
    ChannelNum_ft=[0,250,625,1375,2500,3500,8000];
elseif ChannelNum==8
    SPL_offset = 95.1079128*ones(1,ChannelNum);
    ChannelNum_ft=[0,250,500,750,1375,2500,3500,4875,8000];
elseif ChannelNum==12
    SPL_offset = 95.31420495*ones(1,ChannelNum);
    ChannelNum_ft=[0,250,375,500,750,1125,1500,1875,2625,3375,4250,5625,8000];
elseif ChannelNum==16
    SPL_offset = 96.7119344*ones(1,ChannelNum);
    ChannelNum_ft=[0,250, 375,500,625,750,1000,1250,1625,2000,2375,2875,3500,4250,5125,6125,8000];
end

ua = 0.1;                                                                  
ur = 0.95;                                                                  

FIFO_input = zeros(FFTLen,1);                                              
FIFO_output = zeros(FFTLen-FrameLen,1);                                     

% win = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ...
%        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ...
%        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ...
%        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ...
%        0.0000, 0.0491, 0.0980, 0.1467, 0.1951, 0.2430, 0.2903, 0.3369, ...
%        0.3827, 0.4276, 0.4714, 0.5141, 0.5556, 0.5957, 0.6344, 0.6716, ...
%        0.7071, 0.7410, 0.7730, 0.8032, 0.8315, 0.8577, 0.8819, 0.9040, ...
%        0.9239, 0.9415, 0.9569, 0.9700, 0.9808, 0.9892, 0.9952, 0.9988, ...
%        1.0000, 0.9988, 0.9952, 0.9892, 0.9808, 0.9700, 0.9569, 0.9415, ...
%        0.9239, 0.9040, 0.8819, 0.8577, 0.8315, 0.8032, 0.7730, 0.7410, ...
%        0.7071, 0.6716, 0.6344, 0.5957, 0.5556, 0.5141, 0.4714, 0.4276, ...
%        0.3827, 0.3369, 0.2903, 0.2430, 0.1951, 0.1467, 0.0980, 0.0491, ...
%        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ...
%        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ...
%        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ...
%        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]';  

win = [0.00000000,0.02454123,0.04906767,0.07356456,0.09801714, ...
       0.12241068,0.14673047,0.17096189,0.19509032,0.21910124, ...
       0.24298018,0.26671276,0.29028468,0.31368174,0.33688985, ...
       0.35989504,0.38268343,0.40524131,0.42755509,0.44961133, ...
       0.47139674,0.49289819,0.51410274,0.53499762,0.55557023, ...
       0.57580819,0.59569930,0.61523159,0.63439328,0.65317284, ...
       0.67155895,0.68954054,0.70710678,0.72424708,0.74095113, ...
       0.75720885,0.77301045,0.78834643,0.80320753,0.81758481, ...
       0.83146961,0.84485357,0.85772861,0.87008699,0.88192126, ...
       0.89322430,0.90398929,0.91420976,0.92387953,0.93299280, ...
       0.94154407,0.94952818,0.95694034,0.96377607,0.97003125, ...
       0.97570213,0.98078528,0.98527764,0.98917651,0.99247953, ...
       0.99518473,0.99729046,0.99879546,0.99969882,1.00000000, ...
       0.99969882,0.99879546,0.99729046,0.99518473,0.99247953, ...
       0.98917651,0.98527764,0.98078528,0.97570213,0.97003125, ...
       0.96377607,0.95694034,0.94952818,0.94154407,0.93299280, ...
       0.92387953,0.91420976,0.90398929,0.89322430,0.88192126, ...
       0.87008699,0.85772861,0.84485357,0.83146961,0.81758481, ...
       0.80320753,0.78834643,0.77301045,0.75720885,0.74095113, ...
       0.72424708,0.70710678,0.68954054,0.67155895,0.65317284, ...
       0.63439328,0.61523159,0.59569930,0.57580819,0.55557023, ...
       0.53499762,0.51410274,0.49289819,0.47139674,0.44961133, ...
       0.42755509,0.40524131,0.38268343,0.35989504,0.33688985, ...
       0.31368174,0.29028468,0.26671276,0.24298018,0.21910124, ...
       0.19509032,0.17096189,0.14673047,0.12241068,0.09801714, ...
       0.07356456,0.04906767,0.02454123]';    

audiogram_f  = [125, 250, 500, 1000, 2000, 4000, 8000];                         
audiogram_ht = [HL(1), HL];                                                          
% audiogram_f  = [250, 500, 1000, 2000, 4000, 6000];                          
% audiogram_ht = HL;  


HL_avg = (audiogram_ht(2)+audiogram_ht(3)+audiogram_ht(4)+audiogram_ht(5))/4;
if HL_avg < 60
    HL_Flag = 0;
else
    HL_Flag = 1;
end
% HL_Flag = 1;

if HL_avg > 20 && HL_avg <= 40
    Hearing_Loss = [0 0	0 0 -10 -20 0];
elseif HL_avg > 40 && HL_avg <= 60
    Hearing_Loss = [0 0	-10 -10 -20 -30 -10];
elseif HL_avg > 60
    Hearing_Loss = [0 0	-10 -20 -30 -40 -30];
else 
    Hearing_Loss = [0 0	0 0 0 0 0];
end
Hearing_Loss = 10.^(Hearing_Loss/10);

audiogram_k = audiogram_f./fstep;                                           
x = audiogram_k;
y = Hearing_Loss;

HL_ext = interp1(x, y, (1:64));
TF = isnan(HL_ext); 
HL_ext(TF) = 0;
if HL_Flag==0
    for i=1:length(HL_ext)
        HL_ext(i)=1;
    end
end

[k_n,b_n] = Fit_FIG6Y(audiogram_f, audiogram_ht, ChannelNum);                    
% [k,b] = Fit_FIG6(audiogram_f, audiogram_ht, ChannelNum);

Slen=length(s);                                                             
FrameNum=fix(Slen/FrameLen)-1;                                              
% inSPL_buffer = zeros(ChannelNum);
% splin = zeros(1, FrameNum);
% splout = zeros(1, FrameNum);
inSPL_buffer = zeros(ChannelNum+1);
TK=[40,60];
MPO=110;

% FIFO_outputHL = zeros(FFTLen-FrameLen,1);    

for kk=1:FrameNum

    pos=(kk-1)*FrameLen+1;    
    in=s(pos:pos+FrameLen-1); 
    cur=0;
    gain_dbg = zeros(65,1);
    tmp = zeros(65,1);

    FIFO_input(1:FFTLen-FrameLen)=FIFO_input(FrameLen+1:end);
    FIFO_input(FFTLen-FrameLen+1:end)=in;
    fftss=fft(FIFO_input.*win);
%     HearingLoss_fftss=fft(FIFO_input.*win);

%     if HL_Flag==1
%         for i=1:64
%             HearingLoss_fftss(i)=HearingLoss_fftss(i)*HL_ext(i);
%             HearingLoss_fftss(FFTLen-i)=HearingLoss_fftss(FFTLen-i)*HL_ext(i);
%         end
%     end
    fftss = ones(128,1)./10+ones(128,1)./10*1j;
    HL_ext = ones(1,64);
    for i=2:ChannelNum + 1
        sum=0;

        for j=1:64
            if (125*(cur+j-1))<ChannelNum_ft(i)
                sum=sum+real(fftss(cur+j+1))^2+imag(fftss(cur+j+1))^2;
            end
        end
        spl_in=20*log10(sqrt(sum))+SPL_offset(i-1);    
%         spl_in
%         spl_in = 20*log10(abs(fftss(i)))+ 96; 
        if spl_in > inSPL_buffer(i)
            spl_in = spl_in * (1-ua) + inSPL_buffer(i) * ua; 
        else
            spl_in = spl_in * (1-ur) + inSPL_buffer(i) * ur;
        end
        if spl_in < 0
            spl_in = 0;
        end 
%         spl_in
        inSPL_buffer(i) = spl_in;
       

%         if spl_in < 45
%             spl_out = k(i-1,1)*spl_in+b(i-1,1);
%         elseif spl_in < 75
%             spl_out = k(i-1,2)*spl_in+b(i-1,2);
%         else
%             spl_out = k(i-1,3)*spl_in+b(i-1,3);
%         end
%         if spl_out > 120
%             spl_out = 120;
%         end
        if spl_in < TK(1)
            spl_out = k_n(i-1,1)*spl_in+b_n(i-1,1);
        elseif spl_in < TK(2)&&TK(2)>TK(1)
            spl_out = k_n(i-1,2)*spl_in+b_n(i-1,2);
        else
            spl_out = k_n(i-1,3)*spl_in+b_n(i-1,3);
        end
        if spl_out > MPO
            spl_out = MPO;
        end

        ig = spl_out-spl_in;
        gain = 10^(ig/20);
%         gain_dbg(i-1)=gain;
%         fftss(i) = fftss(i)*gain;
%         fftss(FFTLen-i+2) = fftss(FFTLen-i+2)*gain;
        if ChannelNum==64
            fftss(i) = fftss(i)*gain;
            fftss(FFTLen-i+2) = fftss(FFTLen-i+2)*gain;
        else
            for j=1:64
                if 125*(cur+j-1)<ChannelNum_ft(i)
                    fftss(cur+j+1)=fftss(cur+j+1)*gain*HL_ext(j);
                    gain_dbg(cur+j)=gain;
                    tmp(cur+j)=cur+j+1;
                    if((cur+j)~=FFTLen/2)
                        fftss(FFTLen-(cur+j-1)) = fftss(FFTLen-(cur+j-1))*gain*HL_ext(j);
                    end
                end
            end
        end
        cur=cur+(ChannelNum_ft(i)-ChannelNum_ft(i-1))/125;
    end
    
    fftss(1) = 0;
%     HearingLoss_fftss(1)=0;
    

    ifftss=real(ifft(fftss));
    ifftss=ifftss.*win;
    ifftss(1:FFTLen - FrameLen)=ifftss(1:FFTLen - FrameLen)+FIFO_output;
    FIFO_output=ifftss(FrameLen+1:end);
    

    out(pos:pos+FrameLen-1,1)=ifftss(1:FrameLen);
%     if HL_Flag==1
% 
%         HearingLoss_ifft=real(ifft(HearingLoss_fftss));
%         HearingLoss_ifft=HearingLoss_ifft.*win;
%         %HearingLoss_ifft=HearingLoss_ifft.*32768;
%         HearingLoss_ifft(1:FFTLen - FrameLen)=HearingLoss_ifft(1:FFTLen - FrameLen)+FIFO_outputHL;
%         FIFO_outputHL=HearingLoss_ifft(FrameLen+1:end);
% 
%         outHL(pos:pos+FrameLen-1,1)=HearingLoss_ifft(1:FrameLen);
%     end
end
out=out((FFTLen-FrameLen)+1:end);

end

