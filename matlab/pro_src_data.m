

classdef pro_src_data
    
    properties 
        freq_pattern;
        member_num;
        fs;
        modulation_para;
        modulation;
        carrier;      %% DUC carrier
        src_signal;
        frequency_type;
        protocol_type;
        package_length;
        package_number;
        member_name;
        sample_length;
        package_sample_length;
        slot_length_per;
        ss;
        seg_label;
        seg_label_sort;
    end

    methods
        %% constructor
        %%%   varargin:
        %%%   protocol_type : xx(string)  slot_num:  xxx(number)
        %%%   package_length:   package_number:
        
        %%%   TMDA & slotted ALOHA:  fix slot   
        %%%   CSMA & ALOHA:          not fix slot
        %%%
        function obj = pro_src_data(fs, sample_length, freq_pattern, modulation_para, protocol_type, varargin)
            
        obj.fs = fs;
        obj.member_num = size(fieldnames(modulation_para), 1);
        obj.sample_length = sample_length;
        obj.freq_pattern = freq_pattern;
        obj.protocol_type = lower(protocol_type);
        obj.modulation_para = modulation_para;


        %% get member_name
        obj.member_name = string(fieldnames(obj.modulation_para));
        %% generate empty mapping 
        modulation_map = containers.Map();
        
        %% save modulated signal
        obj.package_length = [];
        obj.package_number = [];

        for i = 1:obj.member_num
            kk = obj.modulation_para.(obj.member_name(i));
            if length(kk.("package_length")) < 0
                throw("need parameter:package_length \n");
            end

            if length(kk.("package_number")) < 0
                throw("need parameter:package_number \n");
            end
            
            obj.package_length = [obj.package_length, kk.("package_length")];
            obj.package_number = [obj.package_number, kk.("package_number")];

        end

        obj.package_sample_length = obj.package_length .* obj.package_number;
        obj.slot_length_per = [];
        
        if obj.protocol_type == "tdma" || obj.protocol_type == "slottedaloha"
            if isempty(varargin)
                thorw("parameter: slot_length is required \n")
            else
                optional_para = cell2mat(varargin);
                optional_para_name = string(fieldnames(cell2mat(varargin)));
                option_para_len = size(optional_para_name, 1);
                obj.slot_length_per = optional_para.(optional_para_name(find(optional_para_name == "slot_length")));
            end

        end

        carrier_map = containers.Map();
        src_signal_map = containers.Map();
        
        for i = 1:obj.member_num
            
            carrier = [];
            
            if size(obj.freq_pattern, 1) .* size(obj.freq_pattern, 2) == 1
                carrier = complex_exponential_wave(obj.freq_pattern * 1e6, obj.fs, obj.package_sample_length(i)).sample_seq;
                % carrier(obj.member_name(i)) = complex_exponential_wave(obj.freq_pattern * 1e6, obj.fs, obj.package_sample_length(i)).sample_seq;
            else
            %% generate different carrier according to frequency pattern
                for j = 1:size(obj.freq_pattern, 2)
                    %% call complex_exponential_wave class to generate carrier
                    %% instantiate complex_exponential_wave class and get member property sample_seq
                    carrier = [carrier; complex_exponential_wave(obj.freq_pattern(i, j) * 1e6, obj.fs, obj.package_length(i)).sample_seq];     
                end
            end

            % carrier_all(i, :) = carrier;
            % carrier = carrier(1:obj.package_sample_length);
            %% save modulation parameter

            if kk.("mod") == "msk"
                %% instantiate msk_modulation class
                modulation_obj = msk_modulation(kk.("symbol_rate"), obj.fs, obj.package_sample_length(i));
                %% call member method to realize DUC
                modulation_obj = modulation_obj.mixer(carrier, 1);
            elseif kk.("mod") == "psk"
                modulation_obj = psk_modulation(kk.("symbol_rate"), obj.fs, obj.package_sample_length(i), kk.("order"), carrier);
            elseif kk.("mod") == "qam"
                modulation_obj = qam_modulation(kk.("symbol_rate"), obj.fs, obj.package_sample_length(i), kk.("order"), carrier);
            end
        
            %% mapping : member_name -> msk_modulation class
            modulation_map(obj.member_name(i)) = modulation_obj;
            carrier_map(obj.member_name(i)) = carrier;
            %% generate fh signal matirx
            
            src_signal_map(obj.member_name(i)) = modulation_obj.modulated_seq_up;

        end

        obj.modulation = modulation_map;
        obj.carrier = carrier_map;

        obj.src_signal = src_signal_map;

        seq = obj.arrange();
        
        obj.seg_label = seq;
        obj.seg_label_sort = sort(seq, 2);

        ss_vector = zeros(obj.member_num, round(1.1 * obj.sample_length));

        if isempty(obj.slot_length_per)

            for i = 1:1:length(obj.member_name)
                seq_temp = [];
                if i == 1
                    seq_temp = sort(seq(:, 1:(obj.package_number(i))), 2);
                else
                    seq_temp = sort(seq(:,               ...
                    sum(obj.package_number(1:i-1)) + 1:  ...
                    sum(obj.package_number(1:i))), 2);
                end
                
                for j = 1:1:size(seq_temp, 2)
                    ss_s = obj.src_signal(obj.member_name(i));
                    if j == 1
                        ss_vector(i, seq_temp(1, j):seq_temp(2, j)) = ss_s(1:max(obj.package_length));

                    else
                        ss_vector(i, seq_temp(1, j):seq_temp(2, j)) = ss_s((j - 1) * max(obj.package_length) : j * max(obj.package_length) - 1);
                    end
                end
            end

        else
            for i = 1:1:length(obj.member_name)
                seq_temp = [];
                if i == 1
                    seq_temp = sort(seq(:, 1:(obj.package_number(i) .* ceil(obj.package_length ./ obj.slot_length_per))), 2);
                else
                    seq_temp = sort(seq(:,                                                                  ...
                    sum(obj.package_number(1:i-1)) .* ceil(obj.package_length ./ obj.slot_length_per) + 1:  ...
                    sum(obj.package_number(1:i)) .* ceil(obj.package_length ./ obj.slot_length_per)), 2);
                end
                
                for j = 1:1:size(seq_temp, 2)
                    ss_s = obj.src_signal(obj.member_name(i));
                    if j == 1
                        ss_vector(i, seq_temp(1, j):seq_temp(2, j)) = ss_s(1:obj.slot_length_per);

                    else
                        ss_vector(i, seq_temp(1, j):seq_temp(2, j)) = ss_s((j - 1) * obj.slot_length_per : j * obj.slot_length_per - 1);
                    end
                end
            end
        end

        obj.ss = sum(ss_vector(:, 1:obj.sample_length));

        end


        function seq = arrange(obj)
            switch obj.protocol_type
            case 'tdma'
                seq = obj.generate_label_table(0);
            case 'slottedaloha'
                seq = obj.generate_label_table(1);
            case 'csma'
                seq = obj.generate_label_table_for_package(0);
            case 'aloha'
                seq = obj.generate_label_table_for_package(1);
            otherwise
                throw("not supported protocol \n")
            end

        end


        function label_table = generate_label_table(obj, occupied_flag)

            package_length2slot_num = ceil(obj.package_length ./ obj.slot_length_per);
            slot_number_in_sample = floor(obj.sample_length ./ obj.slot_length_per);
            % slot_number_in_sample = ceil(obj.sample_length ./ obj.slot_length_per);
            extend_serial = 1.005;
            slot_number_in_sample = round(slot_number_in_sample * extend_serial);

            slot_label = [];

            start_stop_table = [];
            
            if occupied_flag
                slot_label = randi(slot_number_in_sample, [1, sum(package_length2slot_num' .* obj.package_number')]);
            else
                slot_label = randperm(slot_number_in_sample, sum(package_length2slot_num' .* obj.package_number'));
            end

            for i = 1:1:length(slot_label)
                if slot_label(i) == 1
                    start_stop_table(:, i) = [0; 1] .* obj.slot_length_per + [1; 0];
                else
                    start_stop_table(:, i) = [slot_label(i) - 1; slot_label(i)] .* obj.slot_length_per + [0; -1];
                end

            end

            label_table = start_stop_table;

        end


        function label_table_for_package = generate_label_table_for_package(obj, occupied_flag)
            package_len_max = max(obj.package_length);
            package_number_in_sample = floor(obj.sample_length ./ package_len_max);
            extend_serial = 1.005;
            package_number_in_sample = round(package_number_in_sample * extend_serial);

            package_label = [];
            start_stop_table = [];
            if occupied_flag
                package_label = randi(package_number_in_sample, [1, sum(obj.package_number)]);

                for i = 1:1:length(package_label)
                    if package_label(i) == 1
                        start_stop_table(:, i) = [0; 1] .* package_len_max + [1; 0];
                    else
                        start_stop_table(:, i) = [package_label(i) - 1; package_label(i)] .* package_len_max + [0; -1];
                    end
    
                end
            else
                package_label = randperm(package_number_in_sample, sum(obj.package_number));
                serial = 0.5;

                for i = 1:1:length(package_label)
                    if package_label(i) == 1
                        start_stop_table(:, i) = [0; 1] .* package_len_max + [1; 0];
                    else
                        start_stop_table(:, i) = [package_label(i) - 1; package_label(i)] .* package_len_max + [0; -1];
                    end
                end

                a = -1;
                b = 1;
                start_stop_table = sort(start_stop_table, 2);
                if start_stop_table(1, 1) == 1
                    n = sum(obj.package_number) - 1;
                    flag = 1;
                    index = [];
                    while flag
                        r = (b - a) .* rand(n, 1) + a;
                        temp = round(start_stop_table(:, 2:end) + (1 + r') .* min(obj.package_length) .* serial);
                        temp = [start_stop_table(:, 1), temp];
                        for ii = 1:1:size(start_stop_table, 2) - 1
                            index(ii) = temp(1, ii + 1) - temp(2, ii);
                        end
                        if sum(index >= 0) == sum(obj.package_number) - 1
                            flag = 0;
                            start_stop_table = temp;
                        end
                    end

                else
                    n = sum(obj.package_number);
                    flag = 1;
                    index = [];
                    while flag
                        r = (b - a) .* rand(n, 1) + a;
                        temp = round(start_stop_table + (1 + r') .* min(obj.package_length) .* serial);
                        for ii = 1:1:size(start_stop_table, 2) - 1
                            index(ii) = temp(1, ii + 1) - temp(2, ii);
                        end
                        if sum(index >= 0) == sum(obj.package_number) - 1
                            flag = 0;
                            start_stop_table = temp;
                        end
                    end

                end
            end
            

            label_table_for_package = start_stop_table;
        end

    end



end
