
# coding: utf-8

# In[2]:


from scipy import ndimage
from PIL import Image
import numpy as np


# In[22]:


im = np.array(Image.open("/home/kamal/FinalYearProject/FRLibrary-MultipleImage/k.jpg").convert('L'))
k = np.array([[0,1,0],[1,0,1],[0,1,0]])
output = ndimage.convolve(im, k, mode='constant', cval=0.0)
imfile = Image.fromarray(output)
imfile.show()


# In[23]:


output.shape


# In[24]:


im.shape

