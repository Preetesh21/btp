clc;
clear all;
xn = input('Enter the sequence xn = ');  %Get the sequence from user
ln = length(xn);       %find the length of the sequence
N=length(xn);
xn = [xn zeros(1,N-ln)];
xk = zeros(1,N);  %initialize an array of same size as that of input sequence
ixk = zeros(1,N); %initialize an array of same size as that of input sequence
%code block to find the DFT of the sequence
for k = 0:N-1
    for n = 0:N-1
        xk(k+1) = xk(k+1)+(xn(n+1)*exp((-1i)*2*pi*k*n/N));
    end
end
%code block to plot the input sequence
t = 0:N-1;
subplot(2,2,1);
stem(t,xn);
ylabel('Amplitude ---->');
xlabel('n ---->');
title('Input Sequence ---->');
grid on;
magnitude = abs(xk); %Find the magnitudes of individual DFT points
disp('DFT Sequence = ');
disp(magnitude);
%code block to plot the DFT sequence
t = 0:N-1;
subplot(2,2,2);
stem(t,magnitude);
ylabel('Amplitude ---->');
xlabel('K ---->');
title('DFT Sequence ---->');
grid on;
phase = angle(xk); %Find the phases of individual DFT points
disp('Phase = ');
disp(phase);
 %code block to plot the Phase Response
t = 0:N-1;
subplot(2,2,3);
stem(t,phase);
ylabel('Phase ---->');
xlabel('K ---->');
title('Phase Response');
grid on;
% Code block to find the IDFT of the sequence
for n = 0:N-1
    for k = 0:N-1
        ixk(n+1) = ixk(n+1)+(xk(k+1)*exp(1i*2*pi*k*n/N));
    end
end
ixk = ixk./N;
%code block to plot the IDFT sequence
t = 0:N-1;
subplot(2,2,4);
stem(t,ixk);
disp('IDFT Sequence = ');
disp(ixk);
ylabel('Amplitude ---->');
xlabel('n ---->');
title('IDFT sequence ---->');
grid on;
