import cv2 as cv, numpy as np
from numpy.linalg import norm

SZ, CLASS_N = 20, 10
DIGITS_FN = 'digits.png'

def split2d(img, cell_size):
    h, w = img.shape[:2]; sx, sy = cell_size
    cells = [np.hsplit(r, w//sx) for r in np.vsplit(img, h//sy)]
    return np.array(cells).reshape(-1, sy, sx)

def load_digits(fn):
    img = cv.imread(cv.samples.findFile(fn), 0)
    digits = split2d(img, (SZ, SZ))
    labels = np.repeat(np.arange(CLASS_N), len(digits)//CLASS_N)
    return digits, labels

def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2: return img
    skew = m['mu11']/m['mu02']
    M = np.float32([[1,skew,-0.5*SZ*skew],[0,1,0]])
    return cv.warpAffine(img,M,(SZ,SZ),flags=cv.WARP_INVERSE_MAP|cv.INTER_LINEAR)

def preprocess_hog(digits):
    samples=[]
    for img in digits:
        gx,gy=cv.Sobel(img,cv.CV_32F,1,0),cv.Sobel(img,cv.CV_32F,0,1)
        mag,ang=cv.cartToPolar(gx,gy); bin_n=16
        bin=np.int32(bin_n*ang/(2*np.pi))
        hists=[np.bincount(b.ravel(),m.ravel(),bin_n)
              for b,m in zip((bin[:10,:10],bin[10:,:10],bin[:10,10:],bin[10:,10:]),
                             (mag[:10,:10],mag[10:,:10],mag[:10,10:],mag[10:,10:]))]
        hist=np.hstack(hists).astype(np.float32)
        hist/=hist.sum()+1e-7; hist=np.sqrt(hist); hist/=norm(hist)+1e-7
        samples.append(hist)
    return np.float32(samples)

def train_knn(x,y,k=4):
    m=cv.ml.KNearest_create(); m.train(x,cv.ml.ROW_SAMPLE,y); return m
def train_svm(x,y,C=2.67,gamma=5.383):
    m=cv.ml.SVM_create(); m.setKernel(cv.ml.SVM_RBF); m.setType(cv.ml.SVM_C_SVC)
    m.setC(C); m.setGamma(gamma); m.train(x,cv.ml.ROW_SAMPLE,y); return m

def evaluate(model,x,y,name):
    r=model.predict(x)[1].ravel()
    err=(r!=y).mean()*100
    cm=np.zeros((CLASS_N,CLASS_N),np.int32)
    for i,j in zip(y,r): cm[i,int(j)]+=1
    print(f"{name} error: {err:.2f}%\nConfusion matrix:\n{cm}\n")

if __name__=="__main__":
    digits,labels=load_digits(DIGITS_FN)
    shuffle=np.random.permutation(len(digits))
    digits,labels=digits[shuffle],labels[shuffle]
    digits=[deskew(d) for d in digits]; samples=preprocess_hog(digits)
    n=int(0.9*len(samples))
    xtr,xts=np.split(samples,[n]); ytr,yts=np.split(labels,[n])
    evaluate(train_knn(xtr,ytr),xts,yts,"KNN")
    svm=train_svm(xtr,ytr); evaluate(svm,xts,yts,"SVM"); svm.save("digits_svm.dat")
