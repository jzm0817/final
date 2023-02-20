

function names = get_files(path)
    files = dir(path);
    files_num = size(files, 1);
    % names = 0;
    for i = 3:1:files_num
        file_name = string(files(i, 1).name);
        names(:, :, i-2) = file_name;
    end

end