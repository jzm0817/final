%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% class : 
%%%        rx_signal
%%% fea: 
%%%        generate receive signal
%%% parameter: 
%%%    antenna_num: the number of receive antenna
%%%  element_distance: distance between two adjacent antenna
%%%         rx_snr: receive snr
%%%     src_signal: source (multiple) fh signal class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef rx_signal_

    properties

        antenna_num;
        radius;
        rx_snr; 
        src_signal;
        receive_signal;    %% receive signal
        mix_matrix;        
        theta_pattern;
        phi_pattern;     
        freq_pattern;   
        f_pattern;
    end


    methods
        %% constructor
        function obj = rx_signal_(antenna_num, radius, rx_snr, src_signal)
            %% source (multiple) fh signal class
            obj.src_signal = src_signal;
            obj.antenna_num = antenna_num;
            obj.radius = radius;
            obj.rx_snr = rx_snr;
            %% get freq_pattern from source (multiple) fh signal class
            obj.freq_pattern = obj.src_signal.freq_pattern;
            %% get theta_pattern from source (multiple) fh signal class
            obj.theta_pattern = obj.src_signal.theta_pattern;
            obj.phi_pattern = obj.src_signal.phi_pattern;
            obj.mix_matrix = obj.generate_array_structure();
            obj.receive_signal = obj.receive_src_signal();

        end


        %% generate min_matirx according to freq_pattern and theta_pattern

        function mix_matrix = generate_array_structure(obj)
            
            A = [];

            for i = 1:1:size(obj.src_signal.theta_pattern, 2)
                for j = 1:1:size(obj.src_signal.theta_pattern, 1)
                    for k = 1:1:obj.antenna_num
                        tau = (1 / 3e8) * obj.radius * cos(2 * pi * (k - 1) / obj.antenna_num - obj.phi_pattern(j, i) / 180 * pi) * cosd(obj.theta_pattern(j, i));
                        A(k, j, i) = exp(-1j * 2 * pi * obj.src_signal.freq_pattern(j, i) * 1e6 * tau);
                    end
                end

            end
            mix_matrix = A;

        end

        %% generate receive signal according to source signal and mix matrix
        function receive_signal = receive_src_signal(obj)
            %% receive signal matrix
            receive_signal = zeros(obj.antenna_num, obj.src_signal.hop_num * obj.src_signal.hop_length); 
           
            for j = 0:1:obj.src_signal.hop_num - 1
                %% receive signal with agwn
                receive_signal(:, j * obj.src_signal.hop_length + 1: (j+1) * obj.src_signal.hop_length) ...
                    = awgn(obj.mix_matrix(:, :, j+1) * obj.src_signal.src_signal(:, j * obj.src_signal.hop_length + 1: (j+1) * obj.src_signal.hop_length), obj.rx_snr, 'measured');
            
                end

        end

    end

end
