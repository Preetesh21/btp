clc;
%given reactances
Xd=0.3j;  
Xt=0.2j;
XL1=0.3j;
XL2=0.3j;

Pe=0.8; %real power transfered before fault
Q=0.074; %reactive power transfered before fault
S=Pe+Q*j;
V=1;     %infinite bus terminal voltage
I=conj(S)/conj(V); %current flowing into the bus before fault
X1= Xd+Xt+((XL1*XL2)/(XL1+XL2)); %equivalent reactance before fault
E=V+I*X1;  %transient internal voltage of synchronous generator

%prefault condition
Pm=Pe;
P1max= abs(E)*abs(V)/abs(X1); %max value of power delivered by generator in prefault condition
del0= asin(Pe/P1max);

%during fault
X2=(((Xd+Xt)*XL1)+((Xd+Xt)*(XL2/2))+(XL1*(XL2/2)))/0.15j;  %Converting from star to delta to get equivalent reactance during fault.
P2max=abs(E)*abs(V)/abs(X2); %max value of power delivered by generator during fault

%post fault(faulted line outage)
X3=Xd+Xt+XL1;           %L2 line is removed
P3max=abs(E)*abs(V)/abs(X3);  %max value of power delivered by generator after fault

del_max= pi-asin(Pm/P3max);
del_cr=acos((Pm*(del_max-del0)-P2max*cos(del0)+P3max*cos(del_max))/(P3max-P2max))  %critical clearing angle

t0=0;
w0=0;  
f=60;  %frequency=50Hz
H=5;   %inertia constant=5MJ/MVA
M= H/(pi*f);

tn=0.5;  %simulation time
h=0.05;  %time step 
n=(tn-t0)/h; %no. of iterations
t=zeros(1,n+1);  %created an array to store time value after each iteration 
del_1=zeros(1,n+1);
del=zeros(1,n+1);  %created an array to store "del" value after each iteration 
w_1=zeros(1,n+1);
w=zeros(1,n+1);  %created an array to store "w" value after each iteration
t(1)=t0;
del(1)=del0;
w(1)=w0;

fprintf('del(%.2f)=%.4f\n',t(1),del(1));
fprintf('w(%.2f)=%.4f\n',t(1),w(1));

%using for loop for iterations involved in euler method 
for i=1:(((0.1-t0)/h)+1)
    del_1(i+1)= del(i)+h*w(i);
    w_1(i+1)= w(i)+h*((Pm-P2max*sin(del(i)))/M);
    del(i+1)= del(i)+(h/2)*(w(i)+ w_1(i+1));
    w(i+1)= w(i)+(h/2)*(((Pm-P2max*sin(del(i)))/M)+((Pm-P2max*sin(del_1(i+1)))/M));
    t(i+1)=t(i)+h;
        
    fprintf('del(%.2f)=%.4f\n',t(i+1),del(i+1));
    fprintf('w(%.2f)=%.4f\n',t(i+1),w(i+1));
end
%At 0.1 seconds, fault is cleared implies that our operating point will
%shift from P2(during fault condition) to curve P3(after fault condition).
%So, in the loop, we have changed the Pe equation from P2max*sin(del(i)) to
%P3max*sin(del(i)). Then continued the remaining iterations.
for i=1:n
    if(t(i)>0.1)
        del_1(i+1)= del(i)+h*w(i);
        w_1(i+1)= w(i)+h*((Pm-P3max*sin(del(i)))/M);
        del(i+1)= del(i)+(h/2)*(w(i)+ w_1(i+1));
        w(i+1)= w(i)+(h/2)*(((Pm-P3max*sin(del(i)))/M)+((Pm-P3max*sin(del_1(i+1)))/M));
        t(i+1)=t(i)+h;
    
        fprintf('del(%.2f)=%.4f\n',t(i+1),del(i+1));
        fprintf('w(%.2f)=%.4f\n',t(i+1),w(i+1));
    end
end


figure();
plot(t,del);
title('Swing Curve');
xlabel('time(s)');
ylabel('del(rad)');




