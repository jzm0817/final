clear;
close all;

cnt = 1;
if ispc()
    path = "D:/workspace/art/pic/protocol_training/";
elseif isunix()
    path = "/home/jzm/final/pic/protocol_training/";
end

if ~exist(path)
    mkdir(path);
end

% protocol_type = {"TDMA", "ALOHA", "CSMA", "SLOTTEDALOHA"};

protocol_type = {"ALOHA"};
package_len = [1000, 1000, 1000];

mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e6, "package_length", package_len(1), "package_number", 5), ...
"mem1", struct("mod", "psk", "symbol_rate", 5e6, "order", 2, "package_length", package_len(2), "package_number", 5), ...
"mem2", struct("mod", "qam", "symbol_rate", 5e6, "order", 4, "package_length", package_len(3), "package_number", 5));

fs = 610e6;              %%% sample rate
mem_num = size(fieldnames(mod_para), 1);     %%%  get number of fh signal
 
% %%%  return link16 class "l" according to the input parameters
l = link16(mem_num, 5, 0, fs);
% freq = randsample(l.freq_table, 1)
freq = 969;

sample_length = 40000;
slot_len = 1000;
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

% channel = "rayleigh";
channel = "awgn";
snr = 5000;

win_length = 256 * 2;
dft_length = win_length * 2;
win = hann(win_length);
overlap_length = round(0.75 * win_length);

for i = 1:1:size(protocol_type, 2)

    for j = 1:1:cnt
        ss = pro_src_data(fs, sample_length, freq, mod_para, protocol_type{i}, slot_info);
        if isempty(channel)
            src_signal = ss.ss;
        elseif channel == "awgn"
            % eng = norm(ss.ss) ^2 / length(ss.ss);
            % ss.ss = ss.ss / sqrt(eng);
            % norm(ss.ss) ^2 / length(ss.ss)
            % 1 / length(ss.ss) * dot(ss.ss, ss.ss)
            src_signal = awgn(ss.ss, snr, 'measured');
            % size(src_signal)
        elseif channel == "rayleigh"
            src_signal = rayleighchan(ss.ss');
        end
        
        sig_src_tfspec = stft(src_signal, fs, 'FFTLength', dft_length, ...
        'Window', win, 'Centered', false, 'OverlapLength', overlap_length);
        %%% draw source signal   (time-frequency domain)
        % figure('visible', 'off');
        
        % fig = figure;
        contour(abs(sig_src_tfspec(1:win_length, :)));
        % figure;
        % plot(1:1:1000, src_signal(1:1000))
        % figure;
        % axis off;
        % frame = getframe(fig);
        % img = frame2im(frame);
        % imwrite(img, path + protocol_type{i} + '_' + string(j) + '.jpg')
        % saveas(gcf, path + lower(protocol_type{i}) + '_' + string(j) + '.jpg');
    end
end


