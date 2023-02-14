

protocol_type = {"TDMA", "ALOHA", "CSMA", "SLOTTEDALOHA"};
v = [1, 2, 3, 4];
u = repelem(v, 12)
randindex = randperm(size(u, 2))
u = u(randindex)
u = reshape(u, [3, 4 * 4])
pro_cell = []
for i = 1:1:size(u, 2)
    pro_cell = [pro_cell; protocol_type{u(:, i)}]
end
pro_cell(1, :)