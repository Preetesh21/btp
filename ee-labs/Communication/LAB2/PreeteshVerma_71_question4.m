clc;
x=[-1 1 -1 1];
h=[-1 0 2 3 4];
k1=0:length(x)-1;
k2=0:length(h)-1;
subplot(2,2,1);
stem(k1,x);
axis([-1 6 -2 2]);
title('x[n]');
subplot(2,2,2);
stem(k2,h);
axis([-1 6 -2 5]);
title('h[n]');

l1=length(x);
l2=length(h);

%linear convolution
p=l1+l2-1;
x1=[x,zeros(1,(p-l1))];
h1=[h,zeros(1,(p-l2))];

z1=zeros(1,p);

for i=1:p
    for j=1:i
        z1(i)= z1(i)+(x1(j)*h1(i+1-j));
    end;
end;

w1=0:p-1;
subplot(2,2,3)
stem(w1,z1);
axis([0 9 -3 5]);
title('x[n]*h[n] (Linear convolution)');

%circular convolution
N=max(l1,l2);
x2=[x,zeros(1,(N-l1))];
h2=[h,zeros(1,(N-l2))];
z2=zeros(1,N);

for n=1:N
    for m=1:N
        j=mod(n-m,N);
        j=j+1;
        z2(n)=z2(n)+x2(m)*h2(j);
    end;
end;    

w2=0:N-1;
subplot(2,2,4)
stem(w2,z2);
axis([0 7 -5 5]);
title('x[n]*h[n] (Circular convolution)');
