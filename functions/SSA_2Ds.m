%==========================================================================
% Jaime Zabalza, JinchangRen, JiangbinZheng, Junwei Han, Jun Wang,  Huimin Zhao, Shutao Li, and Stephen Marshall.
% Novel Two-Dimensional Singular Spectrum Analysis for Effective Feature Extraction and Data Classification in Hyperspectral Imaging


% Input:
% pic            ！！ The input picture
% u and v        ！！ Embedding window, e.g., u=5 & v=5 is 5*5 window size
% first and last ！！ The number of subcomponents selected,e.g.,
%                     first=1 & last=5 is select 1-5th subcomponents

% Output:
% p_out          ！！ Feature picture after 2D-SSA
%==========================================================================

function [p_out]=SSA_2Ds(pic,u,v,first,last)
[h,w]=size(pic);

%% Embedding (2D)
n=1;
for i=1:1:(h-u+1)
    for j=1:1:(w-v+1)
        temp=pic(i:(i+u-1),j:(j+v-1))';
        col=temp(:);
        X(:,n)=col;
        n=n+1;
    end
end

%% Eigendecomposition
S=X*X';
[U,autoval]=eigs(S,last);
[d,i]=sort(-diag(autoval));
d=-d;
U=U(:,i);
V=(X')*U;

%% grouping
I=[first:last];
Vt=V';
rca=U(:,I)*Vt(I,:);

%% Hankelisation (2D)
p_out=hankel(rca,u,v,h,w);

end


function [y]=hankel(rca,u,v,h,w)
y=hankel_bet(rca,u,v,h,w);
end


function [y]=hankel_bet(rca,u,v,h,w)
% Reconstruction between block function
L=u;
K=h-u+1;
N=h;
y=zeros(N,w);
Lp=min(L,K);
Kp=max(L,K);
for k=0:Lp-2
    for m=1:k+1
        y(k+1,:)=y(k+1,:)+(1/(k+1))*hankel_wit(rca(((m-1)*v+1):(m*v),((k-m+1)*(w-v+1)+1):((k-m+2)*(w-v+1))),v,w); %(m,k-m+2)
    end
end
for k=Lp-1:Kp-1
    for m=1:Lp
        y(k+1,:)=y(k+1,:)+(1/(Lp))*hankel_wit(rca(((m-1)*v+1):(m*v),((k-m+1)*(w-v+1)+1):((k-m+2)*(w-v+1))),v,w); %(m,k-m+2)
    end
end
for k=Kp:N
    for m=k-Kp+2:N-Kp+1
        y(k+1,:)=y(k+1,:)+(1/(N-k))*hankel_wit(rca(((m-1)*v+1):(m*v),((k-m+1)*(w-v+1)+1):((k-m+2)*(w-v+1))),v,w); %(m,k-m+2)
    end
end
end

function [y]=hankel_wit(rca,v,w)
% Reconstruction within block function
L=v;
K=w-v+1;
N=w;
y=zeros(1,N);
Lp=min(L,K);
Kp=max(L,K);
for k=0:Lp-2
    for m=1:k+1
        y(k+1)=y(k+1)+(1/(k+1))*rca(m,k-m+2);
    end
end
for k=Lp-1:Kp-1
    for m=1:Lp
        y(k+1)=y(k+1)+(1/(Lp))*rca(m,k-m+2);
    end
end
for k=Kp:N
    for m=k-Kp+2:N-Kp+1
        y(k+1)=y(k+1)+(1/(N-k))*rca(m,k-m+2);
    end
end
end