%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% class : 
%%%        qam_modulation
%%% fea: 
%%%        qam modulation
%%% parameter: 
%%%    symbol_rate: symbol rate
%%%    sample_rate: sample rate
%%%  sample_length: length of msk modulated signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


classdef qam_modulation

    properties 
        symbol_rate ;
        sample_rate ;
        sample_per_symbol;   %% equal to sample_rate / symbol_rate
        bit_rate ;           %% for msk, bit_rate = symbol_rate
        modulation_type;     %% 'psk'
        sample_length;
        t;                   %% basic time vector
        initial_bit_length;
        symbol_mapping; 
        modulated_seq;       %% base band signal
        modulated_seq_up;    %% frequency band signal
        initial_bit;         %% random symbol
        order;
        rrc_h;
        carrier;
    end

    methods 
        %% constructor
        function obj = qam_modulation(symbol_rate, sample_rate, sample_length, order, carrier, varargin)
            obj.symbol_rate = symbol_rate;
            obj.sample_rate = sample_rate;
            obj.bit_rate = symbol_rate;
            obj.sample_per_symbol = sample_rate / symbol_rate;
            obj.sample_length = sample_length;
            obj.initial_bit_length = ceil(obj.sample_length / obj.sample_per_symbol);
            obj.modulation_type = "qam";
            obj.order = order;
            if ~isempty(varargin)
                obj.rrc_h = cell2mat(varargin);
            else
                obj.rrc_h = rcosdesign(0.7, 4, obj.sample_per_symbol, 'sqrt'); 
            end

            obj.carrier = carrier;
            [obj.modulated_seq, obj.initial_bit, obj.symbol_mapping, obj.modulated_seq_up] = generate_modulated_seq(obj);
            % carrier = complex_exponential_wave(obj.carrier_frequency, obj.sample_rate, obj.sample_length).sample_seq;
            mixer(obj, obj.carrier, 1);
            obj.t = (0:1:obj.sample_length-1) * (1 / obj.sample_rate);
            obj.t = obj.t';
        end

        %% generate base band signal
        function [modulated_seq, initial_bit, symbol_mapping, modulated_seq_up] = generate_modulated_seq(obj)
            %% generate random symbol
            modulate_symbol = randi([0, obj.order - 1], obj.initial_bit_length, 1);
            initial_bit = modulate_symbol;
            %% call function 'qammod' to generate base band modulated signal
            modulated_seq = qammod(modulate_symbol, obj.order);
            symbol_mapping = modulated_seq;
        

            %% cut the base band modulated signal according to parameter 'sample_length'
            modulated_seq_i = upfirdn(real(modulated_seq), obj.rrc_h, obj.sample_per_symbol);
            modulated_seq_q = upfirdn(imag(modulated_seq), obj.rrc_h, obj.sample_per_symbol);
            modulated_seq = modulated_seq_i + 1j * modulated_seq_q;
            modulated_seq = modulated_seq(1:obj.sample_length);
            if obj.initial_bit_length == 1
                modulated_seq = modulated_seq';
            end

            %% base band modulated signal
            % carrier = complex_exponential_wave(obj.carrier_frequency, obj.sample_rate, obj.sample_length).sample_seq;
            modulated_seq_up = real(modulated_seq .* obj.carrier);
            modulated_seq_up = modulated_seq_up ./ max(abs(modulated_seq_up));
        end

        %% DUC
        function obj = mixer(obj, carrier, flag)   

            if flag

                obj.modulated_seq_up = real(obj.modulated_seq .* carrier);
            end

        end

    end
    
end