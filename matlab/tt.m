format long g;
clear;
close all;

%%% parameters of frequency hopping signal  
%%% only support this input format 

% mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e6), ...
%                   "mem1", struct("mod", "msk", "symbol_rate", 5e6), ...
%                   "mem2", struct("mod", "msk", "symbol_rate", 5e6), ...
%                   "mem3", struct("mod", "msk", "symbol_rate", 5e6));

mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e6), ...
                  "mem1", struct("mod", "msk", "symbol_rate", 5e6), ...
                  "mem2", struct("mod", "msk", "symbol_rate", 5e6), ...
                  "mem3", struct("mod", "msk", "symbol_rate", 5e6), ...
                  "mem4", struct("mod", "msk", "symbol_rate", 5e6));

% mod_para = struct("mem0", struct("mod", "msk", "symbol_rate", 5e6), ...
%                   "mem1", struct("mod", "msk", "symbol_rate", 5e6), ...
%                   "mem2", struct("mod", "msk", "symbol_rate", 5e6));

fs = 610e6;              %%% sample rate
hop_period = 76923;      %%% period of frequency hopping signal (hop/s)
hop_length = round(1 / hop_period * fs);  %%% time -> samples
hop_num = 7;             %%% number of hop 
mem_num = size(fieldnames(mod_para), 1);     %%%  get number of fh signal
net_interval = 30;       %%% minimum frequency between two adjacent signal (in MHz) 

%%%  return link16 class "l" according to the input parameters

for i = 1:1:200
    
    l = link16(mem_num, hop_num, net_interval, fs);
end