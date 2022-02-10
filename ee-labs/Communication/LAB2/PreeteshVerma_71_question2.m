% asking the user to enter the two sequences
x=input("Enter input sequence of signal");
h=input("Enter the impulse response");
n1=length(x);% computing the length of the sequences
n2=length(h);
N=max(n1,n2);% length of the output
x=[x,zeros(1,N-n1)];% zero padding the two sequences
h=[h,zeros(1,N-n2)];
y=zeros(1,N);% initializing the output with zeros
% performing convolution
for n=1:N
    for k=1:N
        j=mod(n-k,N);
        j=j+1;
        y(n)= y(n)+(x(k)*h(j));
    end;
end;
% plotting the original sequences,their convolution and in built function convolution
n1=0:length(x)-1;
n2=0:length(h)-1;
subplot(2,2,1);
stem(n1,x);
axis([-1 8 0 2]);
title('x[n]');
subplot(2,2,2);
stem(n2,h);
axis([-1 8 0 2]);
title('h[n]');
w=0:N-1;
subplot(2,2,3)
stem(w,y);
title('x[n]*h[n]');
y2 = cconv(x,h,N);% inbuilt function
w=0:length(y2)-1;
subplot(2,2,4);
stem(w,y2);
title('inbuilt');