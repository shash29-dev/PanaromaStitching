
import numpy as np
import math as m 
from decimal import Decimal
import cv2 as cv

#QUESTION 1(a)
#==============================================================================
def KernelMatrix(k,sigma):
    kernel=np.zeros((k,k),dtype=float)
    def gaussian(i,j):
        return (m.exp(-(i**2 + j**2)/(2*(sigma**2)))/(2*m.pi*(sigma**2)))
    
    def normalize(kernel):
        return (kernel/sum(sum(kernel)));
      
    if k%2 !=0:
        for i in range(m.ceil(-k/2),m.ceil(k/2)):
            for j in range(m.ceil(-k/2),m.ceil(k/2)):
                kernel[i+m.ceil(k/2)-1][j+m.ceil(k/2)-1]= gaussian(i,j)
           
    if k%2 ==0: 
        tmp=m.ceil(-k/2);
        l1=[];
        for i in range(tmp,-tmp+1):
            if i!=0:
                l1.append(i)
        
        for i,x in zip(l1,range(0,k)):
            for j,y in zip(l1,range(0,k)):
                kernel[x][y]=gaussian(i,j)
                
    return normalize(kernel)  

def printKernel(kernel):
    floatFormat = lambda x: '%.3e' % Decimal(x)
    np.set_printoptions(formatter={'float_kind':floatFormat})
    print(kernel)

#QUESTION 1(b)
#==============================================================================
def ImConvolve(I,kernel):
    k=kernel.shape[0]
    [row,col]=I.shape;
    pad=m.floor(k/2);
    Iz=np.zeros((row+2*pad,col+2*pad))
    Iz[pad:Iz.shape[0]-pad,pad:Iz.shape[1]-pad]=I;
#    cv.namedWindow( "PaddedImage", cv.WINDOW_AUTOSIZE );
#    cv.imshow('PaddedImage',np.uint8(Iz))  
    J=np.zeros((I.shape[0],I.shape[1]))
    for i in range(pad,Iz.shape[0]-pad):
        for j in range(pad,Iz.shape[1]-pad):
            imgBlock=Iz[i-pad:i+pad+1,j-pad:j+pad+1];
            convBlock=np.multiply(imgBlock,kernel);
            #convBlock=imgBlock*kernel
            val=sum(sum(convBlock));
            J[i-pad][j-pad]=val; 
    return J;

# QUESTION 2(a)
#==============================================================================
def DoGaussian(k,sigma1,sigma2):
    GaussK1=KernelMatrix(k,sigma1);
    GaussK2=KernelMatrix(k,sigma2);
    DoGK=GaussK2-GaussK1;
    return DoGK;
    
# QUESTION 2(b)
#==============================================================================
def DoGFilter(I,k,sigma1,sigma2):
    kernel=DoGaussian(k,sigma1,sigma2);
    I=ImConvolve(I,k,kernel);
    return I;

# QUESTION 2(c)
#==============================================================================
def ZeroCrossingDet(J):
    for i in range(0,J.shape[0]):
        for j in range(0,J.shape[1]):
            if J[i][j]<0:
                J[i][j]=-1;
            else:
                J[i][j]=1;
    LaplacianKernel=np.array([[1,1,1],
                              [1,-8,1],
                              [1,1,1]]);
    J=ImConvolve(J,k=3,kernel=LaplacianKernel)
    for i in range(0,J.shape[0]):
        for j in range(0,J.shape[1]):
            if J[i][j]<0:
                J[i][j]=0;
            else:
                J[i][j]=1; 
    return J
#==============================================================================
def retConvolved(I,k,kernel):
    BlueCon=ImConvolve(I[:,:,0],kernel=kernel)
    GreenCon=ImConvolve(I[:,:,1],kernel=kernel)
    RedCon=ImConvolve(I[:,:,2],kernel=kernel)
    I_con=np.zeros([I.shape[0],I.shape[1],I.shape[2]])
    I_con[:,:,0]=BlueCon
    I_con[:,:,1]=GreenCon
    I_con[:,:,2]=RedCon
    return I_con

def plotConvolved(I1,I2,I3):
    Out=cv.hconcat((np.uint8(I1), np.uint8(I2),np.uint8(I3)))
    cv.namedWindow('sigma=1,3,20 ConvolvedImage',cv.WINDOW_NORMAL)
    cv.imshow('sigma=1,3,20 ConvolvedImage', Out)
    return Out
    
if __name__ == "__main__":
    kernel1=KernelMatrix(k=9,sigma=1)
    kernel3=KernelMatrix(k=9,sigma=3)
    kernel20=KernelMatrix(k=9,sigma=20)


    I=cv.resize(cv.imread('3_1.jpg'),(256,256))
    cv.namedWindow('OriginalImage',cv.WINDOW_NORMAL);
    cv.imshow('OriginalImage',I)

    I_con1=retConvolved(I,k=9,kernel=kernel1)
    I_con3=retConvolved(I,k=9,kernel=kernel3)
    I_con20=retConvolved(I,k=9,kernel=kernel20)
    cv.waitKey();
    cv.destroyAllWindows()

    Out=plotConvolved(I_con1,I_con3,I_con20)
    cv.waitKey();
    cv.destroyAllWindows()
    
    #------------------------------------------------------------------------------
    I=cv.cvtColor(I,cv.COLOR_BGR2GRAY)
    kernel=DoGaussian(k=11,sigma1=3,sigma2=6);
    print('Answer2(a)- Difference of Gausssian Kernel of Size 11x11:')
    printKernel(kernel);

    #------------------------------------------------------------------------------

    J=DoGFilter(I,k=11,sigma1=2,sigma2=4);
    cv.namedWindow('DoGConvolvedImage',cv.WINDOW_NORMAL)
    cv.imshow('DoGConvolvedImage',J);

    #------------------------------------------------------------------------------

    J=ZeroCrossingDet(J)
    cv.namedWindow('ZeroCrossingDetection',cv.WINDOW_NORMAL)
    cv.imshow('ZeroCrossingDetection',J);
    cv.waitKey();
    cv.destroyAllWindows()
