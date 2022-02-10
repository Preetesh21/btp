close all;
clear all;
i=sqrt(-1);
x_n=input('Give the input of the input sequence:  ');          %Get the sequence from user 
len=length(x_n);                                              %find the length of the sequence


%Plotting the input sequesnce---------------
t=0:len-1;
subplot(221);
stem(t,x_n);
title('Input Sequence');
ylabel ('Amplitude');
xlabel ('Time');

%code for DFT
x_k = fft(x_n);



%Plotting magnitude of DFT---------------------


magnitude=abs(x_k) 
t=0:len-1;
subplot(222);
stem(t,magnitude);
title('DFT Magnitude');
ylabel ('Amplitude');
xlabel ('k');




%Plotting phase of DFT---------------------

phase=angle(x_k)
t=0:len-1;
subplot(223);
stem(t,phase);
title('DFT Phase');
ylabel ('Phase');
xlabel ('k');


% code for calculating idft
inv_x_k=ifft(x_k);




%Plotting magnitude of Inverse of DFT---------------------

t=0:len-1;
subplot(224);
stem(t,inv_x_k);
title('Inverse furior trnsform');
ylabel ('Amplitude');
xlabel ('Time');