
clear;
close all;

save2mat = 1;
index = 9;
pic_number = 5;
multi = 0;
freq_num = 3;
rand_select = 1;
% data_type = "training";
data_type = "test";
cnt = pic_number * freq_num;

prob_vec = [0.2, 0.2, 0.2];
% prob_vec = [0.3];
package_len = [1000, 1000, 1000];
sample_length = 40000;
slot_len = 1000;
slot_info = struct("slot_length", slot_len);
fs = 610e6;

mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e5, "package_length", package_len(1), "package_number", 5), ...
"mem1", struct("mod", "psk", "symbol_rate", 5e5, "order", 2, "package_length", package_len(2), "package_number", 5), ...
"mem2", struct("mod", "qam", "symbol_rate", 5e5, "order", 32, "package_length", package_len(3), "package_number", 5));

if ispc()
    para_path = "D:/workspace/art/data_info_mat/";
elseif isunix()
    para_path = "/home/jzm/workspace/final/data_info_mat/";
end

protocol_type = {"TDMA", "ALOHA", "CSMA", "SLOTTEDALOHA"};
% protocol_type = {"TDMA"};

if multi
    if freq_num ~= size(fieldnames(mod_para), 1)
        fprintf("freq num error\n");
    end
    if  rand_select
        freq = [];
        for i = 1:1:pic_number
            l = link16(freq_num, 1, 30, fs);
            freq = [freq, l.freq_pattern];
        end
    else
        l = link16(freq_num, 1, 30, fs);
        fb_map = l.freq_mapping_base;
        f_index = randperm(size(fb_map, 2), freq_num);
        freq = l.freq_pattern';
    end
else
    if rand_select
        l = link16(freq_num, 1, 30, fs);
        freq = l.freq_pattern';
        if freq_num == 1
            freq = freq(1);
        end
        % freq = l.freq_table(randperm(size(l.freq_table, 2), freq_num)) ;
    else 
        freq = [1203];
    end
end

if ~(freq_num == size(freq, 2)) && multi == 0
    fprintf("freq num error\n");
end

protocol_matrix = [];
if multi
    if cnt < length(protocol_type)
        protocol_matrix = [protocol_matrix, protocol_type{randperm(4, cnt)}];
    else
        v = [1, 2, 3, 4];
        u = repelem(v, cnt / 4);
        randindex = randperm(size(u, 2));
        u = u(randindex);
        u = reshape(u, freq_num, []);
        for i = 1:1:size(u, 2)
            protocol_matrix = [protocol_matrix; protocol_type{u(:, i)}];
        end
    end
end
protocol_matrix = protocol_matrix';


% rayleighchan = comm.RayleighChannel(...
% 'SampleRate',fs, ...
% 'PathDelays',[0 1.5e-6], ...
% 'AveragePathGains',[2 3], ...
% 'NormalizePathGains',true, ...
% 'MaximumDopplerShift',30e2, ...
% 'DopplerSpectrum',{doppler('Gaussian',0.6),doppler('Flat')}, ...
% 'RandomStream','mt19937ar with seed', ...
% 'Seed',22, ...
% 'PathGainsOutputPort',true);
% rayleighchan = comm.RayleighChannel(...
% 'SampleRate',fs, ...
% 'PathDelays',[0 1e-8], ...
% 'AveragePathGains',[1 1.2], ...
% 'NormalizePathGains',true, ...
% 'MaximumDopplerShift',1e3, ...
% 'DopplerSpectrum',{doppler('Gaussian',0.5),doppler('Flat')}, ...
% 'RandomStream','mt19937ar with seed', ...
% 'Seed',22, ...
% 'PathGainsOutputPort',true);


channel = "awgn";
% channel = "rayleigh";
snr = 10;

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