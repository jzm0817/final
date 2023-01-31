clear;
close all;


mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e5), ...
"mem1", struct("mod", "psk", "symbol_rate", 5e5, "order", 2), ...
"mem2", struct("mod", "qam", "symbol_rate", 5e5, "order", 4));

fs = 610e6;              %%% sample rate
hop_period = 76923;      %%% period of frequency hopping signal (hop/s)
hop_length = round(1 / hop_period * fs);  %%% time -> samples
hop_num = 10;             %%% number of hop 
mem_num = size(fieldnames(mod_para), 1);     %%%  get number of fh signal
net_interval = 30;       %%% minimum frequency between two adjacent signal (in MHz) 

%%%  return link16 class "l" according to the input parameters
l = link16(mem_num, hop_num, net_interval, fs);
union_doa = 10;
doa_pattern = repmat((1:1:size(fieldnames(mod_para), 1))' .* union_doa, 1, hop_num);
freq_pattern = l.freq_pattern;
%%%  return fh class "fh_ss" according to the input parameters
%%%  fh_ss contains source frequency hopping signal
fh_ss = fh(fs, mem_num, hop_num, hop_length, net_interval, freq_pattern, doa_pattern, mod_para);


sig_src = sum(fh_ss.src_signal);
% if mem_num == 1
%     sig_src = f_ss.src_signal;
% else
%     sig_src = fh_ss.src_signal(1, :);
%     for i = 2:1:mem_num
%         sig_src = sig_src + fh_ss.src_signal(i, :);
%     end
% end


win_length = 256 * 4;
dft_length = win_length * 2;
win = hann(win_length);
overlap_length = round(0.75 * win_length);

sig_src_tfspec = stft(sig_src, fs, 'FFTLength', dft_length, ...
'Window', win, 'Centered', false, 'OverlapLength', overlap_length);
%%% draw source signal   (time-frequency domain)

contour(abs(sig_src_tfspec(1:win_length, :)))
figure;
contour(abs(sig_src_tfspec))
% clear
% xmin = 0;
% xmax = 40;
% num=5000;  %数据数量
% n=1;    
% h=1;    %均值

% data=zeros(1,num);
% y = @(x,h)(4*x/h).*exp(-2*x/h);
% while n<num
%     x = (xmax-xmin)*rand(1)-xmin;
%     fx=y(x,h);
%     Y = rand(1);
%     if Y<=fx
%         data(1,n)=x;
%         n=n+1;
%     end
% end
% subplot(211);
% stem(data,'filled');title('生成结果')
% subplot(212);
% histogram(data,100);
% hold on
% t=0:0.01:5;
% plot(t,y(t,h)*300,'r','LineWidth',2);xlabel('x');title('四自由度卡方分布')
