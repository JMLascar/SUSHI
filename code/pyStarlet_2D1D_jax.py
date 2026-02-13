#import numpy as np
from copy import deepcopy as dp
import scipy.linalg as lng
import copy as cp
import os
os.environ["JAX_ENABLE_X64"]="True"
from jax import jit
from jax import numpy as jnp

############################################################
################# STARLET TRANSFORM
############################################################

import warnings


class pystarlet:

    # Choose multi-proc or mono at init

    def __init__(self, parallel=False):
        self.parallel = parallel

    def forward(self,array, J=4):
        """
        Wrapper to compute the Starlet transform of a cube of shape (Nz,Nx,Ny)
        Returns an hypercube of shape (Nz,Nx,Ny,nJ)
        """
        if self.parallel is False:
            hypercube=Starlet_Forward3D(array,J=J)

        else :
            hypercube=Starlet_Forward3D_mp(array,J=J)

        return hypercube


class StarletError(Exception):
    """Common `starlet` module's error."""
    pass


class WrongDimensionError(StarletError):
    """Raised when data having a wrong number of dimensions is given.

    Attributes
    ----------
    msg : str
        Explanation of the error.
    """

    def __init__(self, msg=None):
        if msg is None:
            self.msg = "The data has a wrong number of dimension."



##############################################################################

# Need to pad the image symmetrically. 

def pad_image(image,pad):
    return jnp.pad(image,((pad,pad),(pad,pad)),mode="symmetric")
#pad_image=jit(pad_image_jnp,static_argnames=['pad'])

def unpad_image(image,pad):
    return image[pad:-pad,pad:-pad]


def pad_vec(vec,pad):
    #Figure here by how much I need to pad
    return jnp.pad(vec,(pad,pad),mode="symmetric")
#pad_vec=jit(pad_image_jnp,static_argnames=['pad'])

def unpad_vec(vec,pad):
    return vec[pad:-pad]


def smooth_bspline(input_image, step_trou,pad):
    
    coeff_h0 = 3. / 8.
    coeff_h1 = 1. / 4.
    coeff_h2 = 1. / 16.

    step = int(pow(2., step_trou) + 0.5)
    #print("[")
    
    buff=(coeff_h0*input_image[pad:-pad,pad:-pad]+
         coeff_h1*input_image[pad:-pad,pad-step:-pad-step]
         +coeff_h1*input_image[pad:-pad,pad+step:-pad+step]
          +coeff_h2*input_image[pad:-pad,pad-2*step:-pad-2*step]
          +coeff_h2*input_image[pad:-pad,pad+2*step:-pad+2*step]
         )
    buff=pad_image(buff,pad)
    img_out=(coeff_h0*buff[pad:-pad,pad:-pad]+
         coeff_h1*buff[pad-step:-pad-step,pad:-pad]
         +coeff_h1*buff[pad+step:-pad+step,pad:-pad]
          +coeff_h2*buff[pad-2*step:-pad-2*step,pad:-pad]
          +coeff_h2*buff[pad+2*step:-pad+2*step,pad:-pad]
         )
   
    return img_out

def smooth_bspline2D_forcube(input_cube, step_trou,pad):
    
    coeff_h0 = 3. / 8.
    coeff_h1 = 1. / 4.
    coeff_h2 = 1. / 16.

    step = int(pow(2., step_trou) + 0.5)
    #print("[")
    
    buff=(coeff_h0*input_cube[:,pad:-pad,pad:-pad]+
         coeff_h1*input_cube[:,pad:-pad,pad-step:-pad-step]
         +coeff_h1*input_cube[:,pad:-pad,pad+step:-pad+step]
          +coeff_h2*input_cube[:,pad:-pad,pad-2*step:-pad-2*step]
          +coeff_h2*input_cube[:,pad:-pad,pad+2*step:-pad+2*step]
         )
    buff=jnp.pad(buff,((0,0),(pad,pad),(pad,pad)),mode="symmetric")
    img_out=(coeff_h0*buff[:,pad:-pad,pad:-pad]+
         coeff_h1*buff[:,pad-step:-pad-step,pad:-pad]
         +coeff_h1*buff[:,pad+step:-pad+step,pad:-pad]
          +coeff_h2*buff[:,pad-2*step:-pad-2*step,pad:-pad]
          +coeff_h2*buff[:,pad+2*step:-pad+2*step,pad:-pad]
         )
   
    return img_out


def mad(z):
    return jnp.median(jnp.abs(z - jnp.median(z)))/0.6735



def smooth_bspline1D(input_image,step_trou,pad):

    coeff_h0 = 3. / 8.
    coeff_h1 = 1. / 4.
    coeff_h2 = 1. / 16.
    
    step = int(pow(2., step_trou) + 0.5)
    
    vec_out=(coeff_h0*input_vec[pad:-pad]
             +coeff_h1*input_vec[pad-step:-pad-step]
             +coeff_h1*input_vec[pad+step:-pad+step]
             +coeff_h2*input_vec[pad-2*step:-pad-2*step]
             +coeff_h2*input_vec[pad+2*step:-pad+2*step]
            )
    return vec_out

def smooth_bspline1D_forcube(input_vec,step_trou,pad):

#    input_image = np.asarray(input_image,dtype='float64')

    coeff_h0 = 3. / 8.
    coeff_h1 = 1. / 4.
    coeff_h2 = 1. / 16.
    
    step = int(pow(2., step_trou) + 0.5)
    
    vec_out=(coeff_h0*input_vec[pad:-pad,:,:]
             +coeff_h1*input_vec[pad-step:-pad-step,:,:]
             +coeff_h1*input_vec[pad+step:-pad+step,:,:]
             +coeff_h2*input_vec[pad-2*step:-pad-2*step,:,:]
             +coeff_h2*input_vec[pad+2*step:-pad+2*step,:,:]
            )
    return vec_out

def smooth_bspline1D_forhypercube(input_vec,step_trou,pad):

#    input_image = np.asarray(input_image,dtype='float64')

    coeff_h0 = 3. / 8.
    coeff_h1 = 1. / 4.
    coeff_h2 = 1. / 16.
    
    step = int(pow(2., step_trou) + 0.5)
    
    vec_out=(coeff_h0*input_vec[:,pad:-pad,:,:]
             +coeff_h1*input_vec[:,pad-step:-pad-step,:,:]
             +coeff_h1*input_vec[:,pad+step:-pad+step,:,:]
             +coeff_h2*input_vec[:,pad-2*step:-pad-2*step,:,:]
             +coeff_h2*input_vec[:,pad+2*step:-pad+2*step,:,:]
            )
    return vec_out


def Starlet_Forward2D_unjit(input_image,J,M,N):#

    # DO THE WAVELET TRANSFORM #############################################
    input_image=input_image.copy()
    
    wavelet_planes_list = []
    wavelet_planes_list.append(input_image)
    pad=2*int(pow(2., J) + 0.5)+1
    #print(input_image.shape)
    #print(pad)
    num_lines,num_col=M,N
    #wavelet_planes_list=jnp.zeros((M,N,J))

    for scale_index in range(J):
        previous_scale = wavelet_planes_list[scale_index]
        pad_im=pad_image(previous_scale,pad)
        #print(pad_im.shape)
        unpad_im=(smooth_bspline(pad_im,scale_index,pad),pad)
        #print(unpad_im.shape)
        next_scale = smooth_bspline(pad_image(previous_scale,pad), 
            scale_index,pad)

        previous_scale -= next_scale
        wavelet_planes_list[scale_index]=previous_scale

        wavelet_planes_list.append(next_scale)

    coarse = wavelet_planes_list[-1]
    #planes = np.zeros((input_image.shape[0],input_image.shape[1],J))

    planes=jnp.array(wavelet_planes_list[:-1])
    #Need to do this with a loop as numba doesn't accept np.array(wavelet_planes_list[:-1])
    #for i in range(J):
    #    planes[:,:,i]=wavelet_planes_list[i]

    return coarse, planes  #coarse, all other planes
Starlet_Forward2D=Starlet_Forward2D_unjit
#Starlet_Forward2D=jit(Starlet_Forward2D_unjit,static_argnames=['J','M','N'])




def Starlet_Forward2D_for_cube(input_image,J):
    # DO THE WAVELET TRANSFORM ###########################
    input_image=input_image.copy()
    wavelet_planes_list = []
    wavelet_planes_list.append(input_image)
    pad=2*int(pow(2., J) + 0.5)+1

    for scale_index in range(J):
        previous_scale = wavelet_planes_list[scale_index]
        next_scale = smooth_bspline2D_forcube(jnp.pad(previous_scale,
            ((0,0),(pad,pad),(pad,pad)),mode="symmetric"), 
            scale_index,pad)

        previous_scale -= next_scale
        wavelet_planes_list[scale_index]=previous_scale

        wavelet_planes_list.append(next_scale)

    coarse = wavelet_planes_list[-1]
    planes=jnp.array(wavelet_planes_list[:-1])

    return coarse, planes  #coarse, all other planes
#Starlet_Forward2D=jit(Starlet_Forward2D_unjit,static_argnames=['J','M','N'])


def Starlet_Forward1D_unjit(input_image,J,L):#
    # DO THE WAVELET TRANSFORM #############################################
    input_image = input_image.copy()
    wavelet_planes_list = []
    wavelet_planes_list.append(input_image)
    pad=2*int(pow(2., J) + 0.5)+1

    for scale_index in range(J):
        #for sc in range(len(wavelet_planes_list)):
        #    #print(sc," ",wavelet_planes_list[sc])
        previous_scale = wavelet_planes_list[scale_index]

        next_scale = unpad_vec(smooth_bspline1D(pad_vec(previous_scale,pad), scale_index,pad),pad)

        previous_scale -= next_scale
        wavelet_planes_list[scale_index]=previous_scale

        wavelet_planes_list.append(next_scale)

    coarse = wavelet_planes_list[-1]
    planes=jnp.array(wavelet_planes_list[:-1])
    #planes = jnp.zeros((input_image.shape[0],J))

    #for i in range(J):
    #    planes=planes.at[:,i].set(wavelet_planes_list[i])

    return coarse, planes  #coarse, all other planes
Starlet_Forward1D=jit(Starlet_Forward1D_unjit,static_argnames=["J","L"])



def Starlet_Forward1D_forcube(input_image,J):#
    # DO THE WAVELET TRANSFORM #############################################
    input_image = input_image.copy()
    wavelet_planes_list = []
    wavelet_planes_list.append(input_image)
    pad=2*int(pow(2., J) + 0.5)+1

    for scale_index in range(J):
        #for sc in range(len(wavelet_planes_list)):
        #    #print(sc," ",wavelet_planes_list[sc])
        previous_scale = wavelet_planes_list[scale_index]

        next_scale = smooth_bspline1D_forcube(jnp.pad(previous_scale,(
            (pad,pad),(0,0),(0,0)),mode="symmetric"), scale_index,pad)

        previous_scale -= next_scale
        wavelet_planes_list[scale_index]=previous_scale

        wavelet_planes_list.append(next_scale)

    coarse = wavelet_planes_list[-1]
    planes=jnp.array(wavelet_planes_list[:-1])
    return coarse, planes  


def Starlet_Forward1D_forhypercube(input_image,J):#
    # DO THE WAVELET TRANSFORM #############################################
    input_image = input_image.copy()
    wavelet_planes_list = []
    wavelet_planes_list.append(input_image)
    pad=2*int(pow(2., J) + 0.5)+1


    for scale_index in range(J):
        #for sc in range(len(wavelet_planes_list)):
        #    #print(sc," ",wavelet_planes_list[sc])
        previous_scale = wavelet_planes_list[scale_index]

        next_scale = smooth_bspline1D_forhypercube(jnp.pad(previous_scale,(
            (0,0),(pad,pad),(0,0),(0,0)),mode="symmetric"), scale_index,pad)

        previous_scale -= next_scale
        wavelet_planes_list[scale_index]=previous_scale

        wavelet_planes_list.append(next_scale)

    coarse = wavelet_planes_list[-1]
    planes=jnp.array(wavelet_planes_list[:-1])
    return coarse, planes  

#smooth_bspline1D_forcube

def Starlet_Forward2D_1D_unjit(input_image,J_1D=3,J_2D=2):
    #ORDER: J1D, J2D, L,M,N. 
    input_image = input_image.copy()
    c2D,w2D=Starlet_Forward2D_for_cube(input_image,J_2D)
    cc_2D1D,cw_2D1D=Starlet_Forward1D_forcube(c2D,J_1D)
    wc_2D1D,ww_2D1D=Starlet_Forward1D_forhypercube(w2D,J_1D)
    return cc_2D1D,cw_2D1D,wc_2D1D,ww_2D1D

Starlet_Forward2D_1D=jit(Starlet_Forward2D_1D_unjit,static_argnames=["J_1D","J_2D"])
