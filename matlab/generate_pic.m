
function generate_pic(path, protocol_type, package_len, slot_info,       ...
                        mod_para, fs, freq, sample_length,               ...
                        stft_win, stft_dft_length, stft_overlap_length,  ...
                        pic_number, prob_vec, varargin)

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
        label_name = [];
        if size(freq, 2) > 1
            protocol_vec = repmat(protocol_type{i}, 1, size(freq, 2));
            label_name = lower(join(protocol_vec, "-"));
        else
            label_name = lower(protocol_type{i});
        end
        

        for j = 1:1:pic_number
            src_signal = [];
            for k = 1:1:size(freq, 2)
                prob = randsrc(1, 1, [prob_vec; 0.7, 0.2, 0.1]);
                % prob = randsrc(1, 1, [prob_vec; ones(size(prob_vec, 1), size(prob_vec, 2)) / length(prob_vec)]);
                ss = pro_src_data(fs, sample_length, freq(k), mod_para, protocol_type{i}, prob, slot_info);
                % channel, snr
                src_signal(k, :) = ss.ss;
                % if channel == "awgn"
                %     src_signal(k, :) = awgn(ss.ss, snr, 'measured');
                % elseif channel == "rayleigh"
                %     src_signal(k, :) = rayleighchan(ss.ss');
                % end

            end
            if size(freq, 2) > 1
                src_signal = sum(src_signal);
            end
            if channel == "awgn"
                src_signal = awgn(src_signal, snr, 'measured');
            elseif channel == "rayleigh"
                src_signal = rayleighchan(src_signal');
            end

            sig_src_tfspec = stft(src_signal, fs, 'FFTLength', stft_dft_length, ...
            'Window', stft_win, 'Centered', false, 'OverlapLength', stft_overlap_length);
            %%% draw source signal   (time-frequency domain)
            figure('visible', 'off');
            
            % fig = figure;
            contour(abs(sig_src_tfspec(1:length(stft_win), :)));
            fig = gcf;
            % axis off;
            % frame = getframe(fig);
            % img = frame2im(frame);
            % imwrite(img, path + protocol_type{i} + '_' + string(j) + '.jpg')
            saveas(fig, path + label_name + '_' + string(j) + ".jpg");
            clear gcf;
            close all;
            
        end

    end
end