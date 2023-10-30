from skimage import io
import numpy as np
import pandas as pd
from skimage.morphology import disk
import streakfuncs_hough_fast as strf
import sys
from skimage.morphology import binary_dilation
from skimage.morphology import disk
import time


def display_chars(Icentered,exlist,tnum):
    Mask = np.zeros(Icentered.shape)
    for i in range(int(tnum*2)):
        if exlist[i][0] >= Icentered.shape[0]: 
            Mask[Icentered.shape[0]-1,exlist[i][1]] = 1
        elif exlist[i][0] < 0:
            Mask[0,exlist[i][1]] = 1
        elif exlist[i][1] >= Icentered.shape[1]:
            Mask[exlist[i][0],Icentered.shape[1]-1] = 1
        elif exlist[i][1] < 0:
            Mask[exlist[i][0],0] = 1
        else :
            Mask[exlist[i][0],exlist[i][1]] = 1
        d = disk(10)
    Mask = binary_dilation(Mask,d)

    masked = Icentered + Mask*10
    return masked

def fits_analyse(string):
    if string.endswith(".fits"):
        return  1
    else:
        return  0

def streaks(namelist,name,draw):
    #INPUTS #
    # namelist -> list of input images locations "in1.format, in2.format,...."
    # name -> name of the save files
    # draw -> 1 to draw the extremes in an outpur image and save in dir\Results\Imgs\OUT_img_<name>.png'
    start = time.time()
    # Draws the max and tindex images
    names = namelist.split(',')
    fits = 0
    fits = fits_analyse(names[0])

    print('Drawing the max and the TIndex images')
    if fits == 1:
        Iti,t,med = strf.time_index_img_nobin_fits(names)
    else:
        Iti,t,med = strf.time_index_img_nobin(names)
    # application of the TI-filtering methods
    # size can be 2,3 or 4. 2 being the smallest SNR, and 4 the biggest; According to all the testing done until now, 4 has been faster,
    # and when it can't draw the streaks, usually the Next function can's find them using the size = 2
    #alpha \in [0.8,2], two applies a more agressive bright stars filtering
    print('Filtering the TIndex image to find the streaks') 

    b = strf.binary_bright(Iti,2,med)
    tnobrightstars = b*t
    # # Find the position and angles of the streaks 
    # dif = 0.1 is good enough. 
    # delta = 15 seems good, but adjust this values can improve results. And reruning this functions is fast
    # The algorithm finds the streak by running a template
    # which size is 2* delta_i + 1 (width)
    # and height is h_i ( can't be an even number )
    # delta_i must be smaller than delta.
    # Thres is the stoping criteria for the end of the streak
    # thres is the number of pixels that must be zero for the edge of the streak to be
    # considered. 
    # For small SNR, h_i and thres must be high, since the streak can have holes.
    # and delta should also the smallest as possible.
    # thres < (2 x delta_i + 1) x h_i - 1
    print('Finding streak extremes and angles')
    thetas,exs = strf.streak_pos(tnobrightstars,0.1,2,1,3,8) #(t,dif,delta,delta_i,h_i,thres)
    knum = len(exs)//2
    # check if all images have streak candidates
    if knum != np.max(t):
        print('##########################\nOne or more images might not have streaks. \nResults might be flawed.\n##########################')
    ex1 = []
    ex2 = []
    for i in range(int(knum)):
        ex1 = ex1 + [exs[int(2*i)]]
        ex2 = ex2 + [exs[int(2*i+1)]]
    
    exarray = np.array(exs)
    thetareal,notfound = strf.correction_params(exarray,thetas)

    if notfound == 1:
        print('Streaks not found, try to adjust parameters, or colinear streaks might not exist')
    
    print('Saving data.........')
    notfound = strf.save_arrays_to_df(ex1,ex2,thetas,'Results\Data\OUT_data_%s' %name,notfound)
    
    if notfound == 0:
        ex1,ex2 = strf.angle_correction(ex1,ex2,thetareal)
        thaux = []
        for i in range(len(thetas)):
            thaux = thaux + [thetareal] 
        thetas = thaux
        print('The angles of the streaks are: ')
        print(str(thetas))
        print('The 1st extremes of the streaks are: ')
        print(str(ex1)) 
        print('The 2nd extremes of the streaks are: ')
        print(str(ex2)) 

        notfound = strf.save_arrays_to_df(ex1,ex2,thetas,'Results\Data\OUT_data_%s' %name,notfound)
         
    if draw == 1:
        # knum is the number of images used in the  tindex.
        exs=[]
        for i in range(int(knum)):
            exs = exs + [ex1[i],ex2[i]]
        print(exs)
        masked = display_chars(tnobrightstars,exs,knum)
        import matplotlib.image
        matplotlib.image.imsave('Results\Imgs\OUT_img_%s.png'%name, masked)
    end = time.time()
    tempo = end-start
    temp_mins = tempo//60
    secs = tempo % 60 
    print('the time of this operation was ' + str(int(temp_mins)) + ' mins and ' + str(int(np.round(secs))) + ' s.')
    
    return thetas,exs,notfound

#int(sys.argv[6])
if __name__ == "__main__":
    
    streaks(sys.argv[1],str(sys.argv[2]),int(sys.argv[3]))

#(namelist,size,alpha,dif,delta,delta_i,h_i,thres,name,draw)
#
#For low SNR, reduce delta,because noise can corrupt the centroid, and increase h_i also increasing thres to a values close to [(2 x delta+1) x h_i];
# Play with alpha, maybe smaller or equal to 1.5, but not to close to 0.8 due to noise.