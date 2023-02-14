clear 

freq = [1134, 1185, 978];

% freq = sort(freq);
freq_s = sort(freq);
index = [];
for i = 1:1:size(freq, 2)

    for j = 1:1:size(freq_s, 2)
        if freq(i) == freq_s(j)
            index(i) = j;
        end
    end
end