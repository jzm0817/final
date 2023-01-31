
h = rcosdesign(0.7, 4, 610 / 5, 'sqrt');
psk = qam_modulation(5e6, 610e6, 50000, 16, 969e6);

plot(psk.t, psk.modulated_seq_up)
scatterplot(psk.symbol_mapping)