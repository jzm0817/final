


function generate_pic_mul(path, protocol_matrix, package_len, slot_info,  ...
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

    for i = 1:1:pic_number
        src_signal = [];
        if size(freq, 1) > 1
            for k = 1:1:size(freq, 1)
                ss = pro_src_data(fs, sample_length, freq(k, i), mod_para, protocol_matrix(k, i), slot_info);
                src_signal(k, :) = ss.ss;
            end
            src_signal = sum(src_signal);
        else
            for k = 1:1:size(protocol_matrix, 1)
                ss = pro_src_data(fs, sample_length, freq(k), mod_para, protocol_matrix(k, i), slot_info);
                src_signal(k, :) = ss.ss;
            end
            src_signal = sum(src_signal);
        end

        src_signal = reshape(src_signal, 1, []);
        if channel == "awgn"
            src_signal = awgn(src_signal, snr, 'measured');
        elseif channel == "rayleigh"
            src_signal = rayleighchan(src_signal');
        end
        % size(src_signal)
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
        saveas(fig, path + lower(join(protocol_matrix(:, i), "-")) + '_' + string(i) + ".jpg");
        clear gcf;
        close all;
       
    end
end