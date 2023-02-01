clear;
close all;

package_len = [2000, 2000, 2000];
mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e6, "package_length", package_len(1), "package_number", 2), ...
"mem1", struct("mod", "psk", "symbol_rate", 5e6, "order", 2, "package_length", package_len(2), "package_number", 1), ...
"mem2", struct("mod", "qam", "symbol_rate", 5e6, "order", 4, "package_length", package_len(3), "package_number", 3));

fs = 610e6;              %%% sample rate
mem_num = size(fieldnames(mod_para), 1);     %%%  get number of fh signal
 
% %%%  return link16 class "l" according to the input parameters
l = link16(mem_num, 5, 0, fs);
% freq = randsample(l.freq_table, 1)
freq = 969;

sample_length = 40000;

pro_type = "tdma";
% pro_type = "slotted_aloha";
slot_len = 1000;
ss = pro_src_data(fs, sample_length, freq, mod_para, pro_type, struct("slot_length", slot_len));
src_signal = ss.ss;

% rayleighchan = comm.RayleighChannel(...
% 'SampleRate',fs, ...
% 'PathDelays',[0 1.5e-4], ...
% 'AveragePathGains',[2 3], ...
% 'NormalizePathGains',true, ...
% 'MaximumDopplerShift',30e2, ...
% 'DopplerSpectrum',{doppler('Gaussian',0.6),doppler('Flat')}, ...
% 'RandomStream','mt19937ar with seed', ...
% 'Seed',22, ...
% 'PathGainsOutputPort',true);

% src_signal_fade = rayleighchan(ss.ss');

src_signal_awgn = awgn(src_signal, 15);
exist("src_signal_fade")

win_length = 256 * 2;
dft_length = win_length * 2;
win = hann(win_length);
overlap_length = round(0.75 * win_length);

sig_src_tfspec = stft(src_signal_awgn, fs, 'FFTLength', dft_length, ...
'Window', win, 'Centered', false, 'OverlapLength', overlap_length);
%%% draw source signal   (time-frequency domain)

contour(abs(sig_src_tfspec(1:win_length, :)))
% figure off;
% contour(abs(sig_src_tfspec))

