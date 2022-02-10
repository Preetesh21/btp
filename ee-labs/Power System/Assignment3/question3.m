syms V1 V2 V3
eqn1 = 1.11*V1 -V2==0;
eqn2 = 0.11*V1 +1.11*V2 -V3==0;
eqn3 = V1+V2+V3==19.05*10^3;
sol=solve([eqn1,eqn2,eqn3],[V1,V2,V3]);
v1 = sol.V1;
v2 = sol.V2;
v3 = sol.V3;
string_efficiency=(19.05*10^3/(3*v3))*100;
fprintf('The voltages are as %f V %f V %f V \n',v1,v2,v3);
fprintf('String efficiency is %f \n',string_efficiency);