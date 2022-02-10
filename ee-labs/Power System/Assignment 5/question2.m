clc;
% here defining the input matrix given in the question.
params=[1 2 0.2+0.8i 0+0.02i;
2 3 0.3+0.9i 0+0.03i;
2 4 0.25+1i 0+0.04i;
3 4 0.2+0.8i 0+0.02i;
1 3 0.1+0.4i 0+0.01i];

% here we are defining the various variable arrays that would be useful for us in calculation.
start_bus=params(:,1); % starting bus
end_bus=params(:,2); % ending bus
z=params(:,3); % the impedance parameter
y=1./z; %the admittance parameter

bus=max(max(start_bus),max(end_bus));
element=length(start_bus);
Yp=zeros(element+bus,element+bus);
A=zeros(element+bus,bus);
% diagonal matrixx being defined
for i=1:bus
    for j=1:bus
        if(i==j)
            A(i,i)=1;
        end
    end
end
% computing the Y admittance params
for i=bus+1:bus+element
    for j=1:bus
        if(start_bus(i-bus)~=0)
        A(i,start_bus(i-bus))=1;
        end
        if(end_bus(i-bus)~=0)
        A(i,end_bus(i-bus))=-1;
        end
    end
end

B=params(:,4);
for i=1:element
    Yp(i+bus,i+bus)=y(i);
    Yp(start_bus(i),start_bus(i))=Yp(start_bus(i),start_bus(i))+B(i);
    Yp(end_bus(i),end_bus(i))=Yp(end_bus(i),end_bus(i))+B(i);
end
% final calculation
Y=A'*Yp*A;
% displaying the results
display(Y);

M = abs(Y)    %magnitude
Ph = angle(Y) %phase angle

