G=tf([1,0.5,5],[1,5,0,0])
s=tf('s');
C=K*(1+Td/s+Ti*s) ; %Put values of Td and Ti
H=feedback(G*C,1);
controlSystemDesigner('rlocus',H);
stepinfo(H)
damp(H)