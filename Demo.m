%==========================================================================
% H. Fu, et al, "Fusion of PCA and Segmented-PCA Domain Multiscale 2-D-SSA
% for Effective Spectral-Spatial Feature Extraction and Data Classification in Hyperspectral Imagery"

% MSF-PCs on Indian Pines dataset
%==========================================================================

close all;clear all;clc;
addpath(genpath('.\Dataset'));
addpath(genpath('.\libsvm-3.18'));
addpath(genpath('.\functions'));
addpath(genpath('.\drtoolbox'));

%% data
load('indian_pines_gt'); img_gt=indian_pines_gt;
load('Indian_pines_corrected');img=indian_pines_corrected;
[Nx,Ny,bands]=size(img);

tic;
%% Y-PCA
metr=reshape(img,[Nx*Ny,bands]);
D_add=3;
rec_aa=compute_mapping(metr,'PCA',D_add);
YPCA=reshape(rec_aa,[Nx,Ny,D_add]);

%% Y-SPCA
k=10;
YSPCA=SPCA_dim_reduce(img,k);
dim=size(YSPCA,3);

%% Multiscale Spatial Feature Extraction
Lx=[5 10 20 30 40];Ly = Lx;
len=length(Lx);
img2=cell(len,1);
for i=1:len
    img2_i=zeros(Nx,Ny,dim);
    for j=1:dim
        lx=Lx(i);ly=Ly(i);
        img2_i(:,:,j)=SSA_2Ds(YSPCA(:,:,j),lx,ly,1,1);  % 2D-SSA
    end
    img2{i}=img2_i;
    [no_rows,no_lines, no_bands] = size(img2{i});
    img2{i}=reshape(img2{i},[no_rows*no_lines,no_bands]);
    p=8; 
    x=compute_mapping(img2{i},'PCA',p);                 % PCA 
    img2{i}=reshape(x, no_rows,no_lines, p);
    MSF(:,:,i*p-(p-1):i*p)= img2{i};
end
[~,~,B]=size(MSF);

%% fusion
MSF_PCs(:,:,1:D_add)=YPCA;
MSF_PCs(:,:,D_add+1:B+D_add)=MSF;
[~,~,dims]=size(MSF_PCs);

%% training-test samples
Labels=img_gt(:);    
Vectors=reshape(MSF_PCs,Nx*Ny,dims);  
class_num=max(max(img_gt))-min(min(img_gt));
trainVectors=[];trainLabels=[];train_index=[];
testVectors=[];testLabels=[];test_index=[];
rng('default');
Samp_pro=0.02;                                                         %proportion of training samples
for k=1:1:class_num
    index=find(Labels==k);                  
    perclass_num=length(index);           
    Vectors_perclass=Vectors(index,:);    
    c=randperm(perclass_num);                                      
    select_train=Vectors_perclass(c(1:ceil(perclass_num*Samp_pro)),:);    %select training samples
    train_index_k=index(c(1:ceil(perclass_num*Samp_pro)));
    train_index=[train_index;train_index_k];
    select_test=Vectors_perclass(c(ceil(perclass_num*Samp_pro)+1:perclass_num),:); %select test samples
    test_index_k=index(c(ceil(perclass_num*Samp_pro)+1:perclass_num));
    test_index=[test_index;test_index_k];
    trainVectors=[trainVectors;select_train];                    
    trainLabels=[trainLabels;repmat(k,ceil(perclass_num*Samp_pro),1)];
    testVectors=[testVectors;select_test];                      
    testLabels=[testLabels;repmat(k,perclass_num-ceil(perclass_num*Samp_pro),1)];
end
[trainVectors,M,m] = scale_func(trainVectors);
[testVectors ] = scale_func(testVectors,M,m);   
[Vectors ] = scale_func(Vectors,M,m);  

%% SVM-based classification
Ccv=1000; Gcv=0.125;
cmd=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 
models=svmtrain(trainLabels,trainVectors,cmd);
testLabel_est= svmpredict(testLabels,testVectors, models);

%classification map
SVMresult = svmpredict(Labels,Vectors,models); SVMresult = reshape(SVMresult,Nx,Ny);
SVMmap = label2color(SVMresult,'india');figure,imshow(SVMmap);

%classification results
[OA,AA,kappa,CA]=confusion(testLabels,testLabel_est);
result=[CA*100;OA*100;AA*100;kappa*100]
toc;