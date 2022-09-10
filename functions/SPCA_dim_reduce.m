function [ R ]=SPCA_dim_reduce(img,k)
[Nx,Ny,bands]=size(img);

if mod(bands,k)==0
    for i=1:k
        I=reshape(img(:,:,(i-1)*(bands/k)+1:i*(bands/k)),[Nx*Ny,(bands/k)]);
        x=compute_mapping(I,'PCA',1);
        R(:,:,i)=reshape(x,Nx,Ny,1);
    end
else
    b=floor(bands/k);
    for i=1:k
        I=reshape(img(:,:,(i-1)*b+1:i*b),[Nx*Ny,b]);
        x=compute_mapping(I,'PCA',1);
        R(:,:,i)=reshape(x,Nx,Ny,1);
    end
    I1=reshape(img(:,:,(b*k+1):end),[Nx*Ny,rem(bands,k)]);
    x=compute_mapping(I1,'PCA',1);
    R(:,:,k+1)=reshape(x,Nx,Ny,1);
end

end