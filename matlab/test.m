clear;
close all;


mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e6, "package_length", 400, "package_number", 2), ...
"mem1", struct("mod", "psk", "symbol_rate", 5e6, "order", 2, "package_length", 400, "package_number", 1), ...
"mem2", struct("mod", "qam", "symbol_rate", 5e6, "order", 4, "package_length", 400, "package_number", 3));

fs = 610e6;              %%% sample rate
% hop_period = 76923;      %%% period of frequency hopping signal (hop/s)
% hop_length = round(1 / hop_period * fs);  %%% time -> samples
% hop_num = 10;             %%% number of hop 
mem_num = size(fieldnames(mod_para), 1);     %%%  get number of fh signal
% net_interval = 30;       %%% minimum frequency between two adjacent signal (in MHz) 

% %%%  return link16 class "l" according to the input parameters
% l = link16(mem_num, hop_num, net_interval, fs);
% union_doa = 10;
% doa_pattern = repmat((1:1:size(fieldnames(mod_para), 1))' .* union_doa, 1, hop_num);
% freq_pattern = l.freq_pattern;
% %%%  return fh class "fh_ss" according to the input parameters
% %%%  fh_ss contains source frequency hopping signal
% fh_ss = fh(fs, mem_num, hop_num, hop_length, net_interval, freq_pattern, doa_pattern, mod_para);

sample_length = 20000;
freq = 969;
ss = pro_src_data(fs, sample_length, freq, mod_para, "slotted_aloha", struct("slot_length", 200));
% sig_src = sum(ss.src_signal);
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

sig_src_tfspec = stft(ss.ss, fs, 'FFTLength', dft_length, ...
'Window', win, 'Centered', false, 'OverlapLength', overlap_length);
%%% draw source signal   (time-frequency domain)

contour(abs(sig_src_tfspec(1:win_length, :)))
% figure;
% contour(abs(sig_src_tfspec))

