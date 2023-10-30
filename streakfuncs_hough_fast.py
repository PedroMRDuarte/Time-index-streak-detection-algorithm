from skimage import io
import numpy as np
from skimage.morphology import disk
from skimage.filters import threshold_local,threshold_otsu
from skimage.transform import hough_line,hough_line_peaks
from skimage.transform  import rotate as rt
import pandas as pd
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.io.fits import getdata as fitsimg
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground


######################################
## TIME_INDEX FIltering Functions : ##
######################################
def time_index_img_nobin(I_names):
    #Creates the max image (I) and the time index (t)
    I = io.imread(fname= I_names[0],as_gray=True)
    t = np.ones(I.shape[:])
    imlist = []
    imlist.append(I.copy())
    for k in I_names[1:]:
        Iadd = io.imread(fname= k,as_gray=True)
        imlist.append(Iadd.copy())
    if len(I_names)> 2:
        g = np.median(imlist, axis=0).astype(np.uint8)
    else:
        g = np.min(imlist, axis=0).astype(np.uint8)
    t = np.argmax(imlist,axis = 0) + 1 
    z = np.max(imlist, axis=0).astype(np.uint8)

    return z,t,g

def time_index_img_nobin_fits(I_names):
    #Creates the max image (I) and the time index (t)
    image_file = get_pkg_data_filename(I_names[0])
    I = fitsimg(image_file , ext=0)
    t = np.ones(I.shape[:])
    imlist = []
    imlist.append(I.copy())
    for k in I_names[1:]:
        image_file = get_pkg_data_filename(k)
        Iadd = fitsimg(image_file , ext=0)
        imlist.append(Iadd.copy())    
    if len(I_names)> 2:
        g = np.median(imlist, axis=0).astype(np.uint8)
    else:
        g = np.min(imlist, axis=0).astype(np.uint8)    
    t = np.argmax(imlist,axis = 0) + 1      
    z = np.max(imlist, axis=0).astype(np.uint8)   
    return z,t,g

from skimage.morphology import binary_closing as bc
from skimage.morphology import binary_opening as bo

def binarize(I):
    # Uses the background extraction function of astropy to binarize the input image I
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(I, (100, 100), filter_size=(3, 3),
                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    b = (I<(bkg.background_median+2*bkg.background_rms_median))
    return b


def adapt_thresh_open(I,alpha):
    e = binarize(I) 
    ntil = I*e
    T = np.mean(ntil) + alpha * np.std(ntil)
    b = (I>T)
    d = disk(5)
    b = bo(b,d)
    return b

def adapt_thresh_close(I,alpha):
    e = binarize(I)
    ntil = I*e
    T = np.mean(ntil) + alpha * np.std(ntil)
    b = (I>T)
    d = disk(5)
    b = bc(b,d)
    return b


def binary_bright(z,alpha,g):
    #removes bright stars and static highlights
    # INPUTS #
    # z - maximum image
    # alpha -> normalization factor of the binarization (T = mean + \alpha * stdDev ) (suggested value = 2)
    # g -> median image
    
    bmax = adapt_thresh_close(z,alpha)
    bmed = adapt_thresh_open(g,alpha)
    if np.sum(bmed==1) < np.sum(bmax==1):
        if np.sum(bmed==1)<np.sum(bmed==0):
            bhat = bmax*(1-bmed) 
        else:
            bhat = bmax
    else:
        bhat = bmax
    if np.sum(bhat== 1) == 0:
        bhat = bmax
    Ihat = z*bhat
    bhat = adapt_thresh_close(Ihat,alpha)    
    return bhat




############################################
## Extreme and angle tracking Functions : ##
############################################

def find_section(t,dif):
    # This function finds the diretion of the streak
    # Inputs #
    # t -> time index image
    # dif -> minimum distance between two cosecutive directions to evaluate
    # # # # #  
    # Then rotates the image until the streak gets paralel to the x axis of the image, and finds 
    # yl, whicj is the height cooordinate of the streak
    th = np.arange(0,180,dif)
    th = np.deg2rad(th)
    img= t
    out,thetas,d = hough_line(img, theta=th)
    _,angle,dist = hough_line_peaks(out,thetas,d, num_peaks=1)
    an = np.rad2deg(angle[0])
    Irot = rt(img,an)
    Irot = (Irot>0).astype(int)
    (x0,y0) = dist*np.array([np.cos(angle[0]),np.sin(angle[0])])
    n2 = img.shape[1]//2
    x0 = x0-n2
    y0 = y0-n2
    R = np.matrix([[np.cos(np.deg2rad(an)),-np.sin(np.deg2rad(an))],[np.sin(np.deg2rad(an)),np.cos(np.deg2rad(an))]])
    (xl,yl) = np.matmul(R,np.matrix([[y0],[x0]]))
    yl = yl+n2
    theta = -an
    return Irot,yl,theta

def out_center(I):
    # creates a image(nxn)
    # where n is the value the keeps all the t image in the center to be used in the hough
    a,b = I.shape
    r = int(np.sqrt((a/2)**2 + (b/2)**2))
    Iout = np.zeros((r*2, r*2))
    ma = (2*r-a)//2
    mb = (2*r-b)//2
    Iout[ma:a+ma,mb:b+mb] = I 
    return Iout


def centroid(I):
    # I is a binary image (0 or 1)
    #Compute the centroid of the image I (Isection)
    y, x = np.where(I == 1)
    if len(x) == 0 or len(y) == 0:
        x_centroid = 0
        y_centroid= 0
        print("Couldn't find streak - Streak too small")
    else:
        x_centroid = np.mean(x)
        y_centroid = np.mean(y)
    return int(np.round(x_centroid)), int(np.round(y_centroid))

def find_streak_center(Irot,a,delta):
    # Find the streak on the section image
    # I is the tindex image
    # delta is the interval of the section used to find the streak
    # Interval done by [a-delta:a+delta]
    # mx is the maximum of the radon an the predicted size of the streak
    Isection = Irot[:,int(a-delta):int(a+delta)]
    x,y = centroid(Isection)
    if (x == 0 and y == 0):
        print('try to increase the value of delta')
    anew = a+x-delta
    Isection = Irot[:,int(anew-delta):int(anew+delta)]
    # check if the centroid point is one
    if Isection[y,delta] != 1:
         kup = 0
         kdown = 0
         for i in range(Isection.shape[0]-y-1):
             kup =  kup + 1
             if Isection[y+kup,delta] ==1:
                 break
         for i in range(y-1):
             kdown =  kdown + 1
             if Isection[y-kdown,delta] == 1:
                 break
         if kdown < kup:
             y = y - kdown
         else:
             y = y + kup

    return y, anew , Irot.shape[0],Isection



def get_coords(Isec,lr,l,th,al,hc,delta_i,h_i,thres):
    #compute the coordinates of the extremes

    # THis process is done by applying a template from the center of the streak upwards and downwards with the size
    #(2*delta+1 x h_i) until the number of px = 0 is higher than thres
    # lr is the len of the rotated image
    # l is the len of the t expanded image

    #####
    #h_i ---> height of the templ
    # delta ---> width to both sides 
    #template 
    tt = np.ones((h_i,int(2*delta_i+1)))
    #find bl
    upleft = Isec.shape[0]-hc
    centering =  int(Isec.shape[1] - Isec.shape[1]//2 - delta_i)
    haux = 0
    bi = 0
    los = (2*delta_i+1)*h_i-thres

    for i in range(upleft-h_i//2):
        mul = np.multiply(Isec[int(hc+i-h_i//2):int(hc+i+1+h_i//2),centering:centering+2*delta_i+1],tt)
        haux = np.sum(np.reshape(mul,newshape = mul.shape[0]*mul.shape[1]))   
        if haux < los:
            bi = i
            break
        
    #find cl
    ci = 0
    for i in range(hc-h_i//2):
        mul = np.multiply(Isec[int(hc-i-h_i//2):int(hc-i+1+h_i//2),centering:centering+2*delta_i+1],tt) 
        haux = np.sum(np.reshape(mul,newshape = mul.shape[0]*mul.shape[1]))   
        if haux < los:
            ci = i
            break
    # Convert to the out_center image axis
    #bl = hc+bi-h_i//2
    #cl = hc-ci+h_i//2
    fac =((2*delta_i+1)*h_i)//thres
    bl = hc+bi-h_i//fac
    cl = hc-ci+h_i//fac
    rr = lr//2
    r = l//2

    al = al-rr
    bl = bl-rr
    cl = cl-rr
    R = np.array([[np.cos(np.deg2rad(th)),-np.sin(np.deg2rad(th))],[np.sin(np.deg2rad(th)),np.cos(np.deg2rad(th))]])
    R = np.array(np.linalg.inv(np.matrix(R)))
#       
    ex1 = np.matmul(R,np.matrix(np.array([[al],[bl]],dtype=object)))
    ex2 = np.matmul(R,np.matrix(np.array([[al],[cl]],dtype=object)))

#
    ex1 = ex1 + [[r],[r]]
    ex2 = ex2 + [[r],[r]]
    ex1 = [int(ex1[1][0,0]),int(ex1[0][0,0])]
    ex2 = [int(ex2[1][0,0]),int(ex2[0][0,0])]    
    return ex1,ex2

def convert_bin(I,k):
    #Creates a binary image, where 1 corresponds to all point in the image I where the px value is k
    return (I == k).astype(int)

def convert_img_ref(I,Iex,ex1,ex2):
    # converts the coordinates of the extreme back to the referential of the original input image~
    # I -> original sized image
    # Iex -> Extrapolated image
    # ex1,ex2 -> list of extreme points (1 and 2) 
    
    n = I.shape[0]
    m = I.shape[1]
    aout = Iex.shape[0]
    lh = (aout-m)/2
    lw = (aout-n)/2
    ex1 = [int(ex1[0]-lw),int(ex1[1]-lh)]
    ex2 = [int(ex2[0]-lw),int(ex2[1]-lh)]
    return ex1,ex2

def linear_reg(x,y):
    # performs the linear regression and outputs the coefficients.
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.transpose(),x)),x.transpose()),y)

def out_indexExtract(exarray,thetas):
    # extract by angle: 
    # len
    # this function evaluates which points can be considered outliers or not, based on the lenghts of the streaks and their orientation.

    lens = []
    md = np.median(thetas)
    ersth = []
    for i in range(exarray.shape[0]):
        leni = np.sqrt(exarray[i,0]**2 + exarray[i,1]**2)
        lens = lens + [leni]
        if i < exarray.shape[0]//2:
            ersth = ersth + [abs(thetas[i]-md)]
    lens = np.array(lens)
    ersth = np.array(ersth)
    out1 = np.where(lens>=0.1*np.max(lens))[0] 
    out2 = np.where(ersth < 5 )[0] 
    out = np.intersect1d(out1,out2)
    return out
    

def correction_params(exarray,thetas):
    # InPUTS #
    # exarray = np.array( with the extreme coordinates)
    # exarray[:,0] = x - coordinates
    # exarray[:,1] = y - coordinates
    # thetas = list of orientations

    # evaluate outliers(only done for len(thetas) > 2)
    indx = out_indexExtract(exarray,thetas)
    #
    notfound = 0
    x= np.matrix([np.ones(exarray[indx,0].shape),exarray[indx,0]]).transpose()
    y= np.matrix(exarray[indx,1]).transpose()
    b = linear_reg(x,y)
    thetar = float(np.rad2deg(np.arctan(b[1])))
    if thetar > 90:
        thetar = round(thetar - 180,2)
    
    thetarlist = []
    for i in range(len(thetas)):
        thetarlist.append(float(thetar))


    therr = np.abs(np.array(thetas) - np.array(thetarlist))
    idx = np.arange(0,len(therr),1)
    #print('#######\nSTREAKS IN \n' + 'Image idx|' +str(idx)+ '\n' + '1==Found |'+ str((therr<=10).astype(int)) + '\n#######')
    if np.sum(therr >= 10) >=  len(thetas)-1:
        notfound = 1       
    return thetar,notfound
    

def angle_correction(ex1,ex2,thetareal):
    # INPUTS #
    # ex1 -> list of first extremes 
    # ex2 -> list of second extremes
    # thetareal -> corrected theta obtained in the linear regression
    ####
    #OUTPUTS #
    # Lists of the corrected extreme positions ex1c, ex2c
    
    ex1c = []
    ex2c = []
    for i in range(len(ex1)):
        x1 = ex1[i][0]
        y1 = ex1[i][1]
        x2 = ex2[i][0]
        y2 = ex2[i][1]
        ecx = (x1 + x2)/2
        ecy = (y1 + y2)/2
        x1c = x1 - ecx
        x2c = x2 - ecx
        y1c = y1 - ecy
        y2c = y2 - ecy
        # ROTATION
        thetarr = np.deg2rad(thetareal)
        R = np.matrix([[np.cos(thetarr),-np.sin(thetarr)],[np.sin(thetarr),np.cos(thetarr)]])
        Rinv = np.matrix([[np.cos(-thetarr),-np.sin(-thetarr)],[np.sin(-thetarr),np.cos(-thetarr)]])
        ex1R = np.matmul(Rinv,np.matrix([[x1c],[y1c]]))
        ex2R = np.matmul(Rinv,np.matrix([[x2c],[y2c]]))
        ex1p = ex1R[0][0,0]
        ex2p = ex2R[0][0,0]
        ex1R2 = np.matmul(R,np.matrix([[ex1p],[0]]))
        ex2R2 = np.matmul(R,np.matrix([[ex2p],[0]]))
        ex1f = ex1R2 + [[ecx],[ecy]]
        ex2f = ex2R2 + [[ecx],[ecy]]
        ex1f = [int(np.round(ex1f[0][0,0])),int(np.round(ex1f[1][0,0]))]
        ex2f = [int(np.round(ex2f[0][0,0])),int(np.round(ex2f[1][0,0]))] 
        ex1c = ex1c + [ex1f]
        ex2c = ex2c + [ex2f]

    return ex1c,ex2c

###############################
## Hybrid-method Functions : ##
###############################


def streak_pos(t,dif,delta,delta_i,h_i,thres):
    # computes the position of the streak
    #####
    # INPUTS:
    # t -> time index image
    # dif -> minimum distance between two cosecutive directions to evaluate
    # delta -> width parameter of the Isection (Isection_{width} = 2*delta+1) (odd number)
    # delta_i -> width parameter of the search template (STemp_{width} = 2*delta_i+1)  (odd number)
    # h_i -> height parameter of the search template (STemp_{height} = h_i
    # thres -> number ofzero pixels of the search template to stop the search (between 1 and  (2*delta_i+1) * h_i)
    #####
    # outputs a list with angles and extreme coordinates
    # thetalist = [theta1,....,thetaK]
    # exlist = [[ex12,ex21],..., [ex1K,ex2K]]
    kmax = int(np.max(t))
    thetalist = []
    exlist = []
    for k in range(kmax):
        I1 = convert_bin(t,k+1)
        It = out_center(I1)
        l = It.shape[1]
        if np.sum((It==1)/(It.shape[0]*It.shape[1]))<0.05:
            Irot,a,theta= find_section(It,dif)
            center,a,lr,Isec = find_streak_center(Irot,a,delta)
            if center != 0:
                ex1,ex2 = get_coords(Isec,lr,l,theta,a,center,delta_i,h_i,thres)
                ex1,ex2 = convert_img_ref(I1,It,ex1,ex2)
            else:
                ex1,ex2 = [0,0], [0,0]
            exlist= exlist + [ex1,ex2]
            if theta < -90:
                theta = theta + 180
            thetalist = thetalist + [theta]
        else:
            ex1,ex2 = [0,0], [0,0]
            exlist= exlist + [ex1,ex2]
            theta = np.random.rand()*180-90
            if theta < -90:
                theta = theta + 180
            thetalist = thetalist + [theta]
    return thetalist, exlist

def save_arrays_to_df(ex1, ex2, theta, file_name,notfound):
    # Saves the data into a .csv file named <'file_name.csv'>
    # Evaluates the lenghts of the input data and returns not found = 1, if K-1 lenghts are smaller than 2 pixels.
    
    lngt = []
    for i in range(int(len(ex1))):
        l =  np.sqrt((ex1[i][0] - ex2[i][0])**2 + (ex1[i][1] - ex2[i][1])**2)
        lngt = lngt + [l]
    lnf = len([i for i in lngt if i < 5])
    if notfound == 0 and lnf>=(len(theta)-1):
        print("Streaks found are two small and probably noise.")
        notfound = 1        
    df = pd.DataFrame({'ex1': ex1, 'ex2': ex2, 'theta': theta, 'lenght': lngt})
    df.to_csv(f'{file_name}.csv', sep='\t', index=False)
    return notfound