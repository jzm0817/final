
function generate_pic(path, protocol_type, package_len, slot_info,       ...
                        mod_para, fs, freq, sample_length,               ...
                        stft_win, stft_dft_length, stft_overlap_length,  ...
                        pic_number, varargin)

    mem_num = size(fieldnames(mod_para), 1);

    if isempty(varargin)
        snr = 1000;
        channel = "awgn";
    else
        if string(varargin{1}) == "awgn"
            channel = "awgn";
            snr = varargin{2};
        elseif string(varargin{1}) == "rayleigh"
            channel = "rayleigh";
            rayleighchan = varargin{2};
        end
    end

    for i = 1:1:size(protocol_type, 2)

        for j = 1:1:pic_number
            ss = pro_src_data(fs, sample_length, freq, mod_para, protocol_type{i}, slot_info);
            src_signal = [];
            if channel == "awgn"
                src_signal = awgn(ss.ss, snr, 'measured');
            elseif channel == "rayleigh"
                src_signal = rayleighchan(ss.ss');
            end
            sig_src_tfspec = stft(src_signal, fs, 'FFTLength', stft_dft_length, ...
            'Window', stft_win, 'Centered', false, 'OverlapLength', stft_overlap_length);
            %%% draw source signal   (time-frequency domain)
            figure('visible', 'off');
            
            % fig = figure;
            contour(abs(sig_src_tfspec(1:length(stft_win), :)));
            fig = gcf;
            axis off;
            % frame = getframe(fig);
            % img = frame2im(frame);
            % imwrite(img, path + protocol_type{i} + '_' + string(j) + '.jpg')
            saveas(fig, path + lower(protocol_type{i}) + '_' + string(j) + ".jpg");
            clear gcf;
            close all;
        end

    end
end