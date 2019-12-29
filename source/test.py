
import numpy as np
import imutils
import cv2 
import math 
import os
import matplotlib.pyplot as plt
from utils import KernelMatrix
import warnings
from Panaroma import *


if __name__ == "__main__":
    I=ReadAllImages(width=400,Folder='I1',setRandomSeed=True)
    plt.close('all')
    stitch1=PanaromaStitcher()
    stitch1.FindCorrespondeces([I[1],I[0]],Algo='SURF')
    H=stitch1.FindHomographyMatrix()
    result=stitch1.ImageStitcher(stitch='R',clip=False)
    result=blurIfZero(result,kSize=9)
    tmp=cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX);
    tmp=tmp[...,::-1]
    print('----------------------------Stitched 0-1--------------------------------')
    plt.imshow(tmp)
    plt.axis('off')
    plt.title('0+1')
    plt.show()
     
    plt.figure()
    stitch2=PanaromaStitcher()
    stitch2.FindCorrespondeces([I[2],I[1]],Algo='SURF')
    H=stitch2.FindHomographyMatrix()
    result1=stitch2.ImageStitcher(stitch='R',clip=False)
    result1=blurIfZero(result1,kSize=9)
    tmp=cv2.normalize(result1, None, 0, 1, cv2.NORM_MINMAX);
    tmp=tmp[...,::-1]
    print('----------------------------Stitched 1-2--------------------------------')
    plt.imshow(tmp)
    plt.axis('off')
    plt.title('1+2')
    plt.show()

    
    plt.figure()
    stitch3=PanaromaStitcher()
    stitch3.FindCorrespondeces([I[3],result1.astype(np.uint8)])
    H=stitch3.FindHomographyMatrix()
    result2=stitch3.ImageStitcher(stitch='L',clip=False)
    result2=blurIfZero(result2,kSize=9)
    tmp=cv2.normalize(result2, None, 0, 1, cv2.NORM_MINMAX);
    tmp=tmp[...,::-1]
    print('----------------------------Stitched 1-2-3--------------------------------')
    plt.imshow(tmp)
    plt.axis('off')
    plt.title('1+2+3')
    plt.show()

    
    plt.figure()
    stitch4=PanaromaStitcher()
    stitch4.FindCorrespondeces([result2.astype(np.uint8),result.astype(np.uint8)])
    H=stitch4.FindHomographyMatrix()
    resultf=stitch4.ImageStitcher(stitch='L',clip=True)
    resultf=blurIfZero(resultf,kSize=9)
    tmp=cv2.normalize(resultf, None, 0, 1, cv2.NORM_MINMAX);
    tmp=tmp[...,::-1]
    print('----------------------------Stitched 1-2-3-4--------------------------------')
    plt.imshow(tmp)
    plt.axis('off')
    plt.title('1+2+3+4')
    plt.show()


## Inbulit
    print('Generating Panaroma using Inbuilt Functions...')
    result=inbuiltFuncPanaroma(width=400,Folder='I1',setRandomSeed=True)
    plt.figure()
    plt.imshow(result)
    plt.axis('off')
    plt.title('inbuilt Panaroma')
    plt.show()