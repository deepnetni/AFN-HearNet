function ig_new = ig_Update( num,ig,Channel_left_ft )
%IG_ 此处显示有关此函数的摘要
%   此处显示详细说明

Limit_1=2000;
Limit_2=3000;

GainLimith=28;
GainLimitm=24;
GainLimitl=20;

GainLimitHT_l=4;
GainLimitHT_m=6;
GainLimitHT_h=8;

    ig_new=ig;   %初始赋值，若下列皆无合适的 原数值返回
    if num==95
        if Channel_left_ft < Limit_1 && ig >= GainLimitHT_h
            ig_new=GainLimitHT_h;
        else if Channel_left_ft >= Limit_2 && ig > GainLimitHT_l
            ig_new=GainLimitHT_l;
        else if ig > GainLimitHT_m
           ig_new=GainLimitHT_m;
            end
            end
        end
    else
         if Channel_left_ft < Limit_1 && ig >= GainLimith
                ig_new=GainLimith;
         else if Channel_left_ft >= Limit_2 && ig > GainLimitl
                ig_new=GainLimitl;
         else if ig > GainLimitm
                ig_new=GainLimitm;
             end
             end
         end
    end
        
