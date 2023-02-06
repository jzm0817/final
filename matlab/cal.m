
H = 94;
W = 94;

ph = 0;
pw = 0;

sh = 1;
sw = 1;

kh = 5;
kw = 5;

input = [H; W];
p = [ph; pw];
s = [sh; sw];
k = [kh; kw];

output = ((input + 2 * p - k) ./ s) + 1