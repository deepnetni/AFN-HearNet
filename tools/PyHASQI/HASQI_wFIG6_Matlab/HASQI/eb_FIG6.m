function [nalr,delay]=eb_FIG6(HL,nfir,fsamp)
% Function to design an FIR Fig6 equalization filter and a flat filter
% having the same linear-phase time delay.（具有相同线性相位时延的平坦滤波器）
%
% Calling variables:
% HL        Hearing loss at the audiometric frequencies
% nfir		Order of the Fig6 filter and the matching delay（滤波器阶数）
% fsamp     sampling rate in Hz
% level1    optional input specifying level in dB SPL 输入声级（默认为65dB SPL）
%
% Returned arrays:
% nalr		linear-phase filter giving the NAL-R gain function
% delay		pure delay equal to that of the NAL-R filter
%
% James M. Kates, 27 December 2006.
% Version for noise estimation system, 27 Oct 2011.

% Processing parameters
fmax = 0.5*fsamp; %Nyquist frequency（奈奎斯特采样率）

% Audiometric frequencies
aud = [250, 500, 1000, 2000, 4000, 6000]; %Audiometric frequencies in Hz
fv = [0,aud,fmax]; %Frequency vector for the interpolation
cfreq = (0:nfir)/nfir; %Uniform frequency spacing from 0 to 1
HL = interp1(fv,[HL(1),HL,HL(6)],fmax*cfreq);

% Design a flat filter having the same delay as the Fig6 filter
delay = zeros(1,nfir+1);
delay(1+nfir/2) = 1.0;
    
% Design the Fig6 filter for HI listener
gdB = zeros(1,length(HL));
mloss = max(HL); %Test for hearing loss
if mloss > 0
    k = zeros(length(HL),3);                                                  
    b = zeros(length(HL),3);  
    tklin = 45;                                                                
    tkhin = 75;   
%	Compute the Fig6 frequency response at the audiometric frequencies
    for i = 1:length(HL)
        ht = HL(i);
        k(i,1) = 1;
        if ht < 20
            b(i,1) = 0;
            k(i,2) = 1;
            b(i,2) = 0;
            k(i,3) = 1;
            b(i,3) = 0;
        elseif ht < 40
            ig40 = ht - 20;
            splout40 = 40 + ig40;
            b(i,1) = splout40 - 40;
            tklout = tklin + b(i,1);

            ig65 = 0.6 * (ht - 20);
            splout65 = 65 + ig65;
            k(i,2) = (splout65 - tklout) / (65 - tklin);
            b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin);
            tkhout = k(i,2) * tkhin + b(i,2);

            ig95 = 0;
            splout95 = 95 + ig95;
            k(i,3) = (splout95 - tkhout) / (95 - tkhin);
            b(i,3) = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin);
        elseif ht < 60
            ig40 = ht - 20;
            splout40 = 40 + ig40;
            b(i,1) = splout40 - 40;
            tklout = tklin + b(i,1);

            ig65 = 0.6 * (ht - 20);
            splout65 = 65 + ig65;
            k(i,2) = (splout65 - tklout) / (65 - tklin);
            b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin);
            tkhout = k(i,2) * tkhin + b(i,2);

            ig95 = 0.1*(ht - 40)^1.4;
            splout95 = 95 + ig95;
            k(i,3) = (splout95 - tkhout) / (95 - tkhin);
            b(i,3) = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin);
        else
            ig40 = ht - 20 - 0.5 * (ht - 60);
            splout40 = 40 + ig40;
            b(i,1) = splout40 - 40;
            tklout = tklin + b(i,1);

            ig65 = 0.8 * ht - 23;
            splout65 = 65 + ig65;
            k(i,2) = (splout65 - tklout) / (65 - tklin);
            b(i,2) = (tklout * 65 - splout65 * tklin) / (65 - tklin);
            tkhout = k(i,2) * tkhin + b(i,2);

            ig95 = 0.1*(ht - 40)^1.4;
            splout95 = 95 + ig95;
            k(i,3) = (splout95 - tkhout) / (95 - tkhin);
            b(i,3) = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin);
        end
         gdB(i) = k(i,2)*65+b(i,2)-65;      
    end
	gdB=max(gdB,0); %Remove negative gains

%	Design the linear-phase FIR filter
	glin=10.^(gdB/20.0); %Convert gain from dB to linear
	nalr=fir2(nfir,cfreq,glin); %Design the filter (length = nfir+1)
    
else
%	Filters for the normal-hearing subject
	nalr=delay;
end
