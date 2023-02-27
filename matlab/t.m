format long
clear;
close all;

%%% parameters of frequency hopping signal  
%%% only support this input format 
mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e6), ...
                  "mem1", struct("mod", "msk", "symbol_rate", 5e6), ...
                  "mem2", struct("mod", "msk", "symbol_rate", 5e6), ...
                  "mem3", struct("mod", "msk", "symbol_rate", 5e6), ...
                  "mem4", struct("mod", "msk", "symbol_rate", 5e6));

fs = 610e6;              %%% sample rate
hop_period = 76923;      %%% period of frequency hopping signal (hop/s)
hop_length = round(1 / hop_period * fs);  %%% time -> samples
hop_num = 7;             %%% number of hop 
mem_num = size(fieldnames(mod_para), 1);     %%%  get number of fh signal
net_interval = 30;       %%% minimum frequency between two adjacent signal (in MHz) 

%%%  return link16 class "l" according to the input parameters
l = link16(mem_num, hop_num, net_interval, fs);
freq_pattern = l.freq_pattern;   %%% real frequency pattern 

%%%  real doa
union_doa = 5;
doa_pattern = repmat((1:1:size(fieldnames(mod_para), 1))' .* union_doa, 1, hop_num);
union_phi = 15;
phi_pattern = repmat((1:1:size(fieldnames(mod_para), 1))' .* union_phi, 1, hop_num);

%%%  return fh class "fh_ss" according to the input parameters
%%%  fh_ss contains source frequency hopping signal
fh_ss = fh(fs, mem_num, hop_num, hop_length, net_interval, freq_pattern, doa_pattern, phi_pattern, mod_para);

sig_src = [];  %%% save source frequency hopping signal  (vecotr)

if mem_num == 1
    sig_src = f_ss.src_signal;
else
    sig_src = fh_ss.src_signal(1, :);
    for i = 2:1:mem_num
        sig_src = sig_src + fh_ss.src_signal(i, :);
    end
end

ant_num = 4;     %%% number of receive antenna
snr = 12e6;      

win_length = 256;
dft_length = win_length * 2;
win = hann(win_length);
overlap_length = round(0.75 * win_length);


%%%  return rx_signal class "rx"
%%%  rx contains receive signal
rx = rx_signal_(ant_num, 0.1, snr, fh_ss);

th = 0.3;
%%%  return tfdec class "tf"
%%% tf contains estimation
tf = tfdec_(rx, win, overlap_length, dft_length, fs, th, 0);