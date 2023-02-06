
clear;
close all;

save2mat = 1;
index = 0;
pic_number = 2000;
data_type = "training";
% data_type = "test";

if ispc()
    para_path = "D:/workspace/art/data_info_mat/";
elseif isnuix()
    para_path = "/home/jzm/final/data_info_mat/";
end

protocol_type = {"TDMA", "ALOHA", "CSMA", "SLOTTEDALOHA"};
% protocol_type = {"TDMA"};

package_len = [2000, 2000, 2000];

mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e6, "package_length", package_len(1), "package_number", 2), ...
"mem1", struct("mod", "psk", "symbol_rate", 5e6, "order", 2, "package_length", package_len(2), "package_number", 1), ...
"mem2", struct("mod", "qam", "symbol_rate", 5e6, "order", 4, "package_length", package_len(3), "package_number", 3));

fs = 610e6;
freq = 969;

sample_length = 40000;
slot_len = 2000;
slot_info = struct("slot_length", slot_len);

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

% channel = "awgn";
% channel = "rayleigh";
% snr = 1000;

stft_win_length = 256 * 2;
stft_dft_length = stft_win_length * 2;
stft_win = hann(stft_win_length);
stft_overlap_length = round(0.75 * stft_win_length);

if save2mat
    if exist(para_path + data_type + "_para" + string(index) + ".mat")
        delete(para_path + data_type + "_para" + string(index) + ".mat");
        save (para_path + data_type + "_para" + string(index) + ".mat");
    else
        save (para_path + data_type + "_para" + string(index) + ".mat");
    end
end