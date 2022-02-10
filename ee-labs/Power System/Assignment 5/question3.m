clc;
n_bus=4; %number of bus in power system
V=[1.04+0j; 0.98+0.17j; 1+0j; 1.04+0.28j]; %given voltages
Vmag=abs(V);   %array containing magnitude of given bus voltages
delta=angle(V);  %array containing angle(in radian) of given bus voltages

G=[5.88 -2.94 -2.94 -2.94;  
  -2.94 5.88 -2.94 -2.94;
  -2.94 -2.94 5.88 -2.94;
  -2.94 -2.94 -2.94 5.88];  %real part of given admittance

B=[-23.50 11.76 11.76 11.76;
    11.76 -23.50 11.76 11.76;
    11.76 11.76 -23.50 11.76;
    11.76 11.76 11.76 -23.50];    %imaginary part of given admittance


P = zeros(n_bus,1);  %created an array of(4*1)to store real power injected at each bus values
Q = zeros(n_bus,1);  %created an array of(4*1)to store imaginary power injected at each bus values


%for the given iteration,calculating real and imaginary power injected at
%each bus using formula
for i = 1 : n_bus
        for k = 1 : n_bus
            P(i) = P(i) + Vmag(i)*Vmag(k)*(G(i,k)*cos(delta(i)-delta(k)) + B(i,k)*sin(delta(i)-delta(k)));
            Q(i) = Q(i) + Vmag(i)*Vmag(k)*(G(i,k)*sin(delta(i)-delta(k)) - B(i,k)*cos(delta(i)-delta(k)));
        end
end

% The Jacobian matrix

%J11(3*3 matrix)
for i=1:3
       ii=i+1;
       for k=1:3
          kk=k+1;
          j11(i,k)=Vmag(ii)*Vmag(kk)*(G(ii,kk)*sin(delta(ii)-delta(kk))-B(ii,kk)*cos(delta(ii)-delta(kk))); %used formula
       end
       j11(i,i)=-Q(ii)-B(ii,ii)*Vmag(ii)^2; %overwrite the value for(i=k)case.
end

%created an array j211(3*3)from which we have extracted j21 matrix and
%negative of values of j12 for(i not equal to k). 
for i=1:3
      ii=i+1;
       for k=1:3
          kk=k+1;
          j211(i,k)=-Vmag(ii)*Vmag(kk)*(G(ii,kk)*cos(delta(ii)-delta(kk))+B(ii,kk)*sin(delta(ii)-delta(kk))); %formula uesd was same as we do for j21 calculatio.
       end
      j211(i,i)=P(ii)-G(ii,ii)*Vmag(ii)^2;
end

%j21(1*3)
j21= j211(3,1:3); %j21=[M42 M43 M44]= 3rd row of j211 matrix.

%j12(3*1)
j12= -j211(1:3,3); %for i not equal to k case(N42,N43), j12= -3rd column of j211 matrix.
j12(3,1)=P(4)+G(4,4)*Vmag(4)^2; %for i=k case N44, we have overwritten the value using its formula.
   
%j22=j11(1,1);
j22(1,1)=Q(4)-B(4,4)*Vmag(4)^2; %O44


j11
j12
j21
j22
jacobian_matrix=[j11 j12;j21 j22]
   
%All the variables of jacobian matrix are mentioned in theoretical calculations in report.
    
