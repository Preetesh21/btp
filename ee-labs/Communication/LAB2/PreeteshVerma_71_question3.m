
%Linear convolution is the basic operation to calculate the output for any linear time-invariant system given its input and its impulse response.
%Circular convolution is the same thing but considering that the support of the signal is periodic
%length of linear convolution vector is (length(x)+length(h)-1)='11' whereas in
%circular convolution it is max(length(x),length(h))='6'.
%For the given input vectors, linear convolution is linearly increasing, reaching its maximum value
%and then decreasing linearly. On the other hand, we are getting constant value of circular convolution which
%is equal to '6'.
%By using the zero padding, if we make the size of 'x' and 'h' equal to the
%size of linear convolution vector and then apply the circular convolution
%formula, this will give us equal results for both linear and circular
%convolution.