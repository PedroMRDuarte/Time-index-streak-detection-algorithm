import streamlit as st
import pandas as pd
import numpy as np
import TI_hybrid_app as TI
import os
from PIL import Image

import skimage.io as io

def main():
    st.title('Time-index Streak Detection Algorithm')
    st.write('Put the image file locations in the following template: Image1,Image2,Image3,...\n')
    Ims = st.text_input("Image locations:")
    
    Name = st.text_input("Output filename:")
    #Draw = st.text_input("1 == Draw image with extremes")
    
    if st.button("RUN") and Name != '' and Ims != '':
        thetas,exs,masked,nf = TI.streaks(Ims,Name,1)
        dt = pd.DataFrame(data=exs,columns=['x','y'])
        st.table(data = dt)
        st.write('The angle is ' + str(thetas[0]))
        masked = masked*5

        # find zoom level
        ad = 50
        #st.write(dt['x'].values)
        #st.write(type(dt['x'].values))
        xmin = np.min(dt['x'].values) - ad
        xmax = np.max(dt['x'].values) + ad   
        ymin = np.min(dt['y'].values) - ad
        ymax = np.max(dt['y'].values) + ad 

        io.imsave('mtemp.jpg',masked[int(xmin):int(xmax),int(ymin):int(ymax)])     
        m = Image.open('mtemp.jpg')
        st.image(masked,caption = "Extremes")
        st.image(m,caption = "Extremes (zoom)")



main()
