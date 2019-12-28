import numpy as np
import imutils
import cv2 
import math 
import os
import matplotlib.pyplot as plt
from Assignment1_CV import KernelMatrix
import warnings

class PanaromaStitcher():
    def __init__(self):
        pass
#        self.imageA,self.imageB= Images
    
    def FindCorrespondeces(self,ImagePair,Algo='SIFT',ratio=0.75):
        self.imageA,self.imageB=ImagePair
        
        grayImA = cv2.cvtColor(self.imageA,cv2.COLOR_BGR2GRAY)
        grayImB = cv2.cvtColor(self.imageB,cv2.COLOR_BGR2GRAY)
        if Algo=='SURF':
            UseAlgo = cv2.xfeatures2d.SIFT_create()
        else:
            UseAlgo = cv2.xfeatures2d.SIFT_create()
        
        print('Detecting KeyPoints and Features...')
        (kpsA, featuresA) = UseAlgo.detectAndCompute(grayImA, None)
        (kpsB, featuresB) = UseAlgo.detectAndCompute(grayImB, None)
        PointsA = np.float32([keys.pt for keys in kpsA])
        PointsB = np.float32([keys.pt for keys in kpsB])
        MatchBy = cv2.DescriptorMatcher_create('BruteForce')
        
        print('Finding correspondences... Using KNN Matcher')
        MatchesFound= MatchBy.knnMatch(featuresA,featuresB,k=2)
        
        print('Rejecting correspondences below Threshold...')
        GoodMatches=[]
        for m in MatchesFound:
            if len(m)==2 and m[0].distance < m[1].distance*ratio:
                GoodMatches.append((m[0].trainIdx, m[0].queryIdx))
                
        if len(GoodMatches)>4:
            self.ptsA=np.float32([PointsA[i] for (_,i) in GoodMatches])
            self.ptsB=np.float32([PointsB[i] for (i,_) in GoodMatches])
        else:
            print('Matches Found=', len(GoodMatches),'Less Than Four Matches Found.. Homography Cant Be estimated!!')
            
    def FindHomographyMatrix(self,Points=4,inlierP=0.2,SuccessP=0.99,sigma=1):
        
        try:
            n=len(self.ptsA)
        except Exception:
            raise Exception('Find Correspondence first using FindCorrespondences() Method')
            
        if n<4:
            print('Homography Cannot be estimated... Low correspondences!')
            return None
        k=math.log(1-SuccessP)/(math.log(1-math.pow(inlierP,Points)))
        threshold=math.sqrt(5.99)*sigma 
#        T=(1-inlierP)*n
        TotalBestInliersCount=0
        bestH=None
        
        print('Finding Homography...')
        for _ in range(int(k*1.5)):
            try:
                rndIdx=np.random.randint(0,n,size=4)
                cor=[]
                for idx in rndIdx:
                        l1=self.ptsA[idx].tolist()
                        l2=self.ptsB[idx].tolist()
                        cor.append([l1,l2])
                MatrixA=[]
                for l1,l2 in cor:
                    x,y=l1
                    u,v=l2
                    a1=[-x,-y,-1,0,0,0,u*x,u*y,u]
                    a2=[0,0,0,-x,-y,1,v*x,v*y,v]
                    MatrixA.append(a1)
                    MatrixA.append(a2)
                
                MatrixA=np.matrix(MatrixA)    
                U,S,V=np.linalg.svd(MatrixA)    
                H=V[-1].reshape(3,3)
                H=H/H.item(8)
                pts=np.hstack((self.ptsA,np.ones((len(self.ptsA),1))))
                dstPts=[]
                for i in range(pts.shape[0]):
                    dPt=np.asarray(np.matmul(H,pts[i])).squeeze(0)
                    if dPt.item(2)!=0:
                        dPt=(dPt/dPt.item(2)).tolist()
                        dstPts.append(dPt)
                    else:
                        self.ptsB=np.delete(self.ptsB,i)
                        self.ptsA=np.delete(self.ptsA,i)
                PointsStack=np.hstack((self.ptsB,np.array(dstPts)[:,0:2]))
            #    diff=sum(np.square(PointsStack[:,2]-PointsStack[:,0]))+sum(np.square(PointsStack[:,3]-PointsStack[:,1]))
                diffa=np.expand_dims(np.square(PointsStack[:,2]-PointsStack[:,0]),axis=1)
                diffb=np.expand_dims(np.square(PointsStack[:,3]-PointsStack[:,1]),axis=1)
                diff=np.hstack((diffa,diffb))
                error=np.expand_dims(np.linalg.norm(diff,axis=1),axis=1)
                inliers=(error<threshold).sum()
                
                if  inliers>TotalBestInliersCount:
                    TotalBestInliersCount=inliers
                    bestH=H
#                    if TotalBestInliersCount>T:
#                        break
            except Exception:
                pass
            
        print('Found H with maximum inlier Counts=',TotalBestInliersCount)
        self.H=bestH
        return self.H
    
    def ImageStitcher(self,stitch='L',clip=True):
        if stitch=='L':
            self.WarpedIm=self.StitchToLeft(clip=clip)
        else:
            self.WarpedIm=self.StitchToRight(clip=clip)
        return self.WarpedIm
            
    def StitchToLeft(self,clip):
        print('Stitching Left...')
        h,w,_=self.imageA.shape
        inH,inW,_=self.imageB.shape
        
        x=np.linspace(0,w,w,endpoint=False)
        y=np.linspace(0,h,h,endpoint=False)
        Xx,Xy=np.meshgrid(x,y,indexing='xy')
        Points=np.vstack((Xx.ravel(),Xy.ravel()))
        dstPts=[]
        for i in range(Points.shape[1]):
            pts=np.array([Points[0][i],Points[1][i],1])
            dPt=np.asarray(np.matmul(self.H,pts)).squeeze(0)
            dPt=(dPt/dPt.item(2)).tolist()
            dstPts.append(dPt)
        
        dstPts=np.array(dstPts)
        dstPts=np.vstack((Points,dstPts[:,0],dstPts[:,1])).astype(int)
        ImDim=[min(dstPts[2]),min(dstPts[3]),max(dstPts[2]),max(dstPts[3])]
        inImDim=[0,0,inW,inH]
        Dim=np.vstack((inImDim,ImDim))
        maxmin=np.vstack((np.max(Dim,axis=0), np.min(Dim,axis=0)))
        height=max(maxmin[:,3])-min(maxmin[:,1])
        width=max(maxmin[:,2])-min(maxmin[:,0])
        try:
            Im=np.zeros((height+2,width+2,3))
            shifty=min(maxmin[:,1])
            Im[(Dim[0][1]-shifty):(Dim[0][3]-shifty),0:self.imageB.shape[1],:]=self.imageB
            Im[dstPts[3,:]-shifty,dstPts[2,:]]=self.imageA[dstPts[1,:],dstPts[0,:]]
        except Exception :
            print('Try Failed...!!!')
            warnings.warn("H transformed co-ordinates to large values... results likely to be Wrong!  Proceeding...")
            Im=np.zeros((int(inH*1.5),int(inW*1.5),3))
            Im[(Dim[0][1]):(Dim[0][3]),0:self.imageB.shape[1],:]=self.imageB
            for i in range(len(dstPts[0])):
                try:
                    Im[dstPts[3,i],dstPts[2,i]]=self.imageA[dstPts[1,i],dstPts[0,i]]
                except Exception :
                    pass
        if clip:
            row=[]
            col=[]
            for i in range(Im.shape[0]):
                if list(Im[i,:,1]).count(0)>inW:
                    row.append(i)
            for i in range(Im.shape[1]):
                if list(Im[:,i,1]).count(0)>int(inH/2):
                    col.append(i)
            Im=np.delete(Im,row,axis=0)
            Im=np.delete(Im,col,axis=1)
        if Im.size==0:
            print('Clipping Returned NoneType Image.. setting clip=False internally ')
            Im=self.StitchToLeft(clip=False)
        return Im
    
    def StitchToRight(self,clip=True):
        print('Stitching Right...')
        h,w,_=self.imageB.shape
        inH,inW,_=self.imageA.shape
        
        inIm=self.imageA
        x=np.linspace(0,w,w,endpoint=False)
        y=np.linspace(0,h,h,endpoint=False)
        Xx,Xy=np.meshgrid(x,y,indexing='xy')
        Points=np.vstack((Xx.ravel(),Xy.ravel()))
        dstPts=[]
        for i in range(Points.shape[1]):
            pts=np.array([Points[0][i],Points[1][i],1])
            dPt=np.asarray(np.matmul(np.linalg.inv(self.H),pts)).squeeze(0)
            if dPt.item(2)!=0:
                dPt=(dPt/dPt.item(2)).tolist()
                dstPts.append(dPt)
                
        dstPts=np.array(dstPts)
        dstPts=np.vstack((Points,dstPts[:,0],dstPts[:,1])).astype(int)
        ImDim=[min(dstPts[2]),min(dstPts[3]),max(dstPts[2]),max(dstPts[3])]
        inImDim=[0,0,inW,inH]
        Dim=np.vstack((inImDim,ImDim))
        maxmin=np.vstack((np.max(Dim,axis=0), np.min(Dim,axis=0)))
        height=max(maxmin[:,3])-min(maxmin[:,1])
        width=max(maxmin[:,2])-min(maxmin[:,0])
        dsx=dstPts[2,:]-Dim[1,0]
        dsy=dstPts[3,:]-Dim[1,1]
        shifty=min(maxmin[:,1])
        try:
            Im=np.zeros((height+2,width+2,3))
            Im[(Dim[0][1]-shifty):(Dim[0][3]-shifty),Im.shape[1]-inIm.shape[1]:Im.shape[1],:]=inIm
            Im[dsy,dsx]=self.imageB[dstPts[1,:],dstPts[0,:]]
        except Exception:
            print('Try Failed...!!!')
            warnings.warn("H transformed co-ordinates to large values... results likely to be Wrong!  Proceeding...")

            Im=np.zeros((int(inH*1.5),int(inW*1.5),3))
            Im[(Dim[0][1]):(Dim[0][3]),Im.shape[1]-inIm.shape[1]:Im.shape[1],:]=inIm
            for i in range(len(dstPts[0])):
                try:
                    Im[dsy,dsx]=self.imageB[dstPts[1,:],dstPts[0,:]]  
                except Exception:
                    pass
        if clip:
            row=[]
            col=[]
            for i in range(Im.shape[0]):
                if list(Im[i,:,1]).count(0)>inW:
                    row.append(i)
            for i in range(Im.shape[1]):
                if list(Im[:,i,1]).count(0)>int(inH/2):
                    col.append(i)
            Im=np.delete(Im,row,axis=0)
            Im=np.delete(Im,col,axis=1)
        if Im.size==0:
            print('Clipping Returned NoneType Image.. setting clip=False internally ')
            Im=self.StitchToRight(clip=False)
        return Im
                 
def ReadAllImages(width=400,Folder='I1',setRandomSeed=True):
         #I1,I2,I4,I5 I3(50G,55), I6(30)
    if setRandomSeed:
        if Folder=='I1' or Folder=='I2' or Folder=='I4' or Folder=='I5':
            np.random.seed(seed=10)
        elif Folder=='I3':
            np.random.seed(seed=55)
    ImageF=os.path.abspath(os.path.join(os.getcwd(),"..",'Images/'+Folder))
    I=cv2.imread(ImageF+'/STA_0031.JPG',0)
    I=[]
    ImageName=os.listdir(ImageF)
    seq=[]
    seqDict={}
    for i in ImageName:
	    seq.append(i[-8:-4])
	    seqDict[i[-8:-4]]=i
    seq=sorted(seq) 
    print('Check images are in sequence, Sequence read is as :', seq)
    for s in seq:
	    img=cv2.imread(ImageF+"/"+seqDict[s])
	    img=imutils.resize(img, width=width)
	    I.append(img)
    return I

def ImConvolve(I,kernel):
    k=kernel.shape[0]
    [row,col]=I.shape;
    pad=math.floor(k/2);
    Iz=np.zeros((row+2*pad,col+2*pad))
    Iz[pad:Iz.shape[0]-pad,pad:Iz.shape[1]-pad]=I;
    J=np.zeros((I.shape[0],I.shape[1]))
    for i in range(pad,Iz.shape[0]-pad):
        for j in range(pad,Iz.shape[1]-pad):
            if Iz[i,j]==0:
                imgBlock=Iz[i-pad:i+pad+1,j-pad:j+pad+1];
                convBlock=np.multiply(imgBlock,kernel);
                val=sum(sum(convBlock));
                J[i-pad][j-pad]=val; 
            else:
                J[i-pad][j-pad]=Iz[i,j]
    return J;

def blurIfZero(Im,kSize=11):
    
# Use of this fun slowers the code
    kernel= KernelMatrix(k=kSize,sigma=1)
    channel1=ImConvolve(Im[:,:,0],kernel)
    channel2=ImConvolve(Im[:,:,1],kernel)
    channel3=ImConvolve(Im[:,:,2],kernel)
    J=np.stack((channel1,channel2,channel3),axis=2)
    return J

def inbuiltFuncPanaroma(width=400,Folder='I1',setRandomSeed=True):
    reprojThresh=2
    def Stitch(result,Image):
        for i in range(Image.shape[0]):
            for j in range(Image.shape[1]):
                if result[i,j,0]==0:
                    result[i,j,:]=Image[i,j,:]
        return result
            
    I=ReadAllImages(width=width,Folder=Folder,setRandomSeed=setRandomSeed)
    stitch1=PanaromaStitcher()
    stitch1.FindCorrespondeces([I[1],I[0]],Algo='SURF')
    (H, status) = cv2.findHomography(stitch1.ptsA, stitch1.ptsB, cv2.RANSAC, reprojThresh)
    result = cv2.warpPerspective(stitch1.imageA, H,(stitch1.imageA.shape[1] + stitch1.imageB.shape[1], stitch1.imageA.shape[0]))
    result= Stitch(result,stitch1.imageB)
    
    stitch1=PanaromaStitcher()
    stitch1.FindCorrespondeces([I[2],result],Algo='SURF')
    (H, status) = cv2.findHomography(stitch1.ptsA, stitch1.ptsB, cv2.RANSAC, reprojThresh)
    result = cv2.warpPerspective(stitch1.imageA, H,(stitch1.imageA.shape[1] + stitch1.imageB.shape[1], stitch1.imageA.shape[0]))
    result= Stitch(result,stitch1.imageB)
    
    stitch1=PanaromaStitcher()
    stitch1.FindCorrespondeces([I[3],result],Algo='SURF')
    (H, status) = cv2.findHomography(stitch1.ptsA, stitch1.ptsB, cv2.RANSAC, reprojThresh)
    result = cv2.warpPerspective(stitch1.imageA, H,(stitch1.imageA.shape[1] + stitch1.imageB.shape[1], stitch1.imageA.shape[0]))
    result= Stitch(result,stitch1.imageB)
    
    stitch1=PanaromaStitcher()
    stitch1.FindCorrespondeces([I[4],result],Algo='SURF')
    (H, status) = cv2.findHomography(stitch1.ptsA, stitch1.ptsB, cv2.RANSAC, reprojThresh)
    result = cv2.warpPerspective(stitch1.imageA, H,(stitch1.imageA.shape[1] + stitch1.imageB.shape[1], stitch1.imageA.shape[0]))
    result= Stitch(result,stitch1.imageB)
    return result[...,::-1]
 
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