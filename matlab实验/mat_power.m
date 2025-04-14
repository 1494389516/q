function F=mat_power(A,k)
[V,T]=jordan(A);
vec=diag(T);
V1=[0,diag(T,1)',0];
V2=find(V1==0);
lam=vec(V2(1:end-1));
m=length(lam);
for i =1:m
    k0=v2(i):(V2(i+1)-1);
    J1=T(K0,k0);
    F(k0,k0)=powerJ(J1,k);
end
F=simplify(V*F*inv(V));
end
