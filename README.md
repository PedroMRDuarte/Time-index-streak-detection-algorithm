# Time-index-streak-detection-algorithm

This folder has two versions of the TI streak detection algorithm.
THe first version needs to be run on a GPU to be time efficient and uses the files:
- **TI_hybrid.py**
- **streakfuncs_hough_cuda.py**
- Folder Results with two folders inside called Imgs and Data

Thee secound version is the Faster vaariation:
- **TI_hybrid_fast.py**
- **streakfuncs_hough_fast.py**
- Folder Results with two folders inside called Imgs and Data

  **This two variations run by calling the following prompt in on a python console in the file directory:**
  _python TI_hybrid<_fast>.py "file1dir,file2dir,...,fileKdir" "savename" (draw)_

  _draw_ == 1 means it saves an image with the extremes marked in it

The files **TI_hybrid_app.py** and **TI_app.py** run the fast algorithm on a app powered by streamlit, and can be called by running the following code: 

_streamlit run TI_app.py_


**Requirements:**
- Python 3.10.9;
- scikit-image 0.19.3;
- numpy 1.24.3;
- numba 0.57.0;
- astropy 5.2.1;
- photutils 1.8.0;
- matplotlib 3.6.2;
- pandas 1.5.3;
- streamlit 1.24.1(optional).
