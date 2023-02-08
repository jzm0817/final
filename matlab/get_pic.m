
% clear;
% close all;

t_type = "protocol";
data_type = "test";

file_number = 5;

if ispc()
    para_path = "D:/workspace/art/data_info_mat/";
    h5file_path = "D:/workspace/art/data_h5/";
    if ~exist(para_path)
        thorw("parameter dictory not exist \n");
    end
elseif isunix()
    para_path = "/home/jzm/workspace/final/data_info_mat/";
    h5file_path = "/home/jzm/workspace/final/data_h5/";
    if ~exist(para_path)
        thorw("parameter dictory not exist \n");
    end
end

file_name = get_files(para_path);

if  exist("file_number")
    vec = file_number:1:file_number
else
    vec = 1:1:size(file_name, 3);
end

for i = vec

    tt = strsplit(file_name(i), '.');
    para_info = tt{1};

    if ispc()
        save_pic_path= "D:/workspace/art/pic/protocol/" + t_type  + "_" + para_info; 
    elseif isunix()
        save_pic_path = "/home/jzm/workspace/final/pic/protocol/" + t_type +  "_" + para_info;
    end

    if exist(save_pic_path)
        rmdir(save_pic_path, 's');
        save_pic_path = save_pic_path + "/";
        mkdir(save_pic_path);
    else
        save_pic_path = save_pic_path + "/";
        mkdir(save_pic_path);
    end

    load(para_path + file_name(i));

    if exist(h5file_path + t_type + data_type + "_" + para_info + ".hdf5")
        delete(h5file_path + t_type + data_type + "_" + para_info + ".hdf5");
    end
    if exist("channel") ~= 1
        channel = "awgn";
        snr = 1000;
    end

    if channel == "awgn"
        channel_info = snr;
    elseif channel == "rayleigh"
        channel_info = rayleighchan;
    end

    generate_pic(save_pic_path, protocol_type, package_len, slot_info, ...
                mod_para, fs, freq, sample_length,...
                stft_win, stft_dft_length, stft_overlap_length, ...
                pic_number, channel, channel_info);

end

