function y = diff_eq( A,B,y0,U,d )
% A,B- Coefficient of difference equation
% y0-Initial vector
% U-Z-transform of input signal 
% d-Delay steps
E=0;n=length(A)-1;syms z;
if nargin==4
d=0;
end
m=length(B)-1;u=iztrans(U);u0=subs(u,0:m-1);
for i=1:n
E=E+A(i)*y0(1:n+1-i)*(z.^(n+1-i:-1:1)).'; %(5-7-5)
end
for i=1:m
E=E-B(i)*u0(1:m+1-i)*(z.^(m+1-i:-1:1)).'; %(5-7-6)
end
Y=(poly2sym(B,z)*U*z^(-d)+E)/poly2sym(A,z); %(5-7-4)
y=iztrans(Y);
