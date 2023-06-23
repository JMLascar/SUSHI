# Functions
import numpy as np
from copy import deepcopy as dp
import scipy.linalg as lng
import copy as cp
import warnings

##############################################################################
# STARLET TRANSFORM
# The starlet transform part of this file was adapted from
# Starlet transform ref: Starck, J.-L. & Murtagh, F. 1994,
# Astronomy and Astrophysics, 342
##############################################################################


try:
    from numba import njit, prange
    print("Numba imported")
except ModuleNotFoundError:
    warnings.warn("Cannot use Numba. Switch to low performance mode.")
    # Make a decorator that does nothing if numba cannot be used
    def njit(f,parallel=False,fastmath=False):
        return f

    def prange():
        return range


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

@njit(parallel=False, fastmath = True)
def get_pixel_value(image, x, y, type_border):

    if type_border == 0:

        #try:
        pixel_value = image[x, y]
        return pixel_value
        #except IndexError as e:
        #    return 0

    elif type_border == 1:

        num_lines, num_col = image.shape    # TODO
        x = x % num_lines
        y = y % num_col
        pixel_value = image[x, y]
        return pixel_value

    elif type_border == 2:

        num_lines, num_col = image.shape    # TODO

        if x >= num_lines:
            x = num_lines - 2 - x
        elif x < 0:
            x = abs(x)

        if y >= num_col:
            y = num_col - 2 - y
        elif y < 0:
            y = abs(y)

        pixel_value = image[x, y]
        return pixel_value

    elif type_border == 3:

        num_lines, num_col = image.shape    # TODO

        if x >= num_lines:
            x = num_lines - 1 - x
        elif x < 0:
            x = abs(x) - 1

        if y >= num_col:
            y = num_col - 1 - y
        elif y < 0:
            y = abs(y) - 1

        pixel_value = image[x, y]
        return pixel_value

    else:
        raise ValueError()


@njit(parallel=False, fastmath = True)
def smooth_bspline(input_image, type_border, step_trou):
    """Apply a convolution kernel on the image using the "à trou" algorithm.

    Pseudo code:

    **convolve(scale, $s_i$):**

    $c_0 \leftarrow 3/8$

    $c_1 \leftarrow 1/4$

    $c_2 \leftarrow 1/16$

    $s \leftarrow \lfloor 2^{s_i} + 0.5 \rfloor$

    **for** all columns $x_i$

    $\quad$ **for** all rows $y_i$

    $\quad\quad$ scale[$x_i$, $y_i$] $\leftarrow$ $c_0$ . scale[$x_i$, $y_i$] + $c_1$ . scale[$x_i-s$, $y_i$] + $c_1$ . scale[$x_i+s$, $y_i$] + $c_2$ . scale[$x_i-2s$, $y_i$] + $c_2$ . scale[$x_i+2s$, $y_i$]

    **for** all columns $x_i$

    $\quad$ **for** all rows $y_i$

    $\quad\quad$ scale[$x_i$, $y_i$] $\leftarrow$ $c_0$ . scale[$x_i$, $y_i$] + $c_1$ . scale[$x_i$, $y_i-s$] + $c_1$ . scale[$x_i$, $y_i+s$] + $c_2$ . scale[$x_i$, $y_i-2s$] + $c_2$ . scale[$x_i$, $y_i+2s$]

    Inspired by Sparse2D mr_transform (originally implemented in *isap/cxx/sparse2d/src/libsparse2d/IM_Smooth.cc* in the
    *smooth_bspline()* function.

    ```cpp
    void smooth_bspline (const Ifloat & Im_in,
                         Ifloat &Im_out,
                         type_border Type, int Step_trou) {
        int Nl = Im_in.nl();  // num lines in the image
        int Nc = Im_in.nc();  // num columns in the image
        int i,j,Step;
        float Coeff_h0 = 3. / 8.;
        float Coeff_h1 = 1. / 4.;
        float Coeff_h2 = 1. / 16.;
        Ifloat Buff(Nl,Nc,"Buff smooth_bspline");

        Step = (int)(pow((double)2., (double) Step_trou) + 0.5);

        for (i = 0; i < Nl; i ++)
        for (j = 0; j < Nc; j ++)
           Buff(i,j) = Coeff_h0 *    Im_in(i,j)
                     + Coeff_h1 * (  Im_in (i, j-Step, Type)
                                   + Im_in (i, j+Step, Type))
                     + Coeff_h2 * (  Im_in (i, j-2*Step, Type)
                                   + Im_in (i, j+2*Step, Type));

        for (i = 0; i < Nl; i ++)
        for (j = 0; j < Nc; j ++)
           Im_out(i,j) = Coeff_h0 *    Buff(i,j)
                       + Coeff_h1 * (  Buff (i-Step, j, Type)
                                     + Buff (i+Step, j, Type))
                       + Coeff_h2 * (  Buff (i-2*Step, j, Type)
                                     + Buff (i+2*Step, j, Type));
    }
    ```

    Parameters
    ----------
    input_image
    type_border
    step_trou

    Returns
    -------

    """

#    input_image = np.asarray(input_image,dtype='float64')

    coeff_h0 = 3. / 8.
    coeff_h1 = 1. / 4.
    coeff_h2 = 1. / 16.

    num_lines, num_col = input_image.shape    # TODO

#    buff = np.zeros(input_image.shape, dtype='float64')
#    img_out = np.zeros(input_image.shape, dtype='float64')

    buff = np.zeros_like(input_image)
    img_out = np.zeros_like(input_image)

    step = int(pow(2., step_trou) + 0.5)

    for i in range(num_lines):
        for j in range(num_col):
            buff[i,j]  = coeff_h0 *    get_pixel_value(input_image, i, j,        type_border)
            buff[i,j] += coeff_h1 * (  get_pixel_value(input_image, i, j-step,   type_border) \
                                     + get_pixel_value(input_image, i, j+step,   type_border))
            buff[i,j] += coeff_h2 * (  get_pixel_value(input_image, i, j-2*step, type_border) \
                                     + get_pixel_value(input_image, i, j+2*step, type_border))

    for i in range(num_lines):
        for j in range(num_col):
            img_out[i,j]  = coeff_h0 *    get_pixel_value(buff, i,        j, type_border)
            img_out[i,j] += coeff_h1 * (  get_pixel_value(buff, i-step,   j, type_border) \
                                        + get_pixel_value(buff, i+step,   j, type_border))
            img_out[i,j] += coeff_h2 * (  get_pixel_value(buff, i-2*step, j, type_border) \
                                        + get_pixel_value(buff, i+2*step, j, type_border))

    return img_out



@njit(parallel=False,fastmath = True)
def Starlet_Forward2D(input_image,J=4):
    """Compute the starlet transform of `input_image`.

    Pseudo code:

    **wavelet_transform(input_image, num_scales):**

    scales[0] $\leftarrow$ input_image

    **for** $i \in [0, \dots, \text{num_scales} - 2]$

    $\quad$ scales[$i + 1$] $\leftarrow$ convolve(scales[$i$], $i$)

    $\quad$ scales[$i$] $\leftarrow$ scales[$i$] - scales[$i + 1$]


    Inspired by Sparce2D mr_transform (originally implemented in *isap/cxx/sparse2d/src/libsparse2d/MR_Trans.cc*)

    ```cpp
    static void mr_transform (Ifloat &Image,
                              MultiResol &MR_Transf,
                              Bool EdgeLineTransform,
                              type_border Border,
                              Bool Details) {
        // [...]
        MR_Transf.band(0) = Image;
        for (s = 0; s < Nbr_Plan -1; s++) {
           smooth_bspline (MR_Transf.band(s),MR_Transf.band(s+1),Border,s);
           MR_Transf.band(s) -= MR_Transf.band(s+1);
        }
        // [...]
    }
    ```

    Parameters
    ----------
    input_image : array_like
        The input image to transform.
    J : int, optional
        The number of scales used to transform `input_image` or in other words
        the number of wavelet planes returned.
    Returns
    -------
    list
        Return a list containing the wavelet planes.

    Raises
    ------
    WrongDimensionError
        If `input_image` is not a 2D array.
    """

#    input_image = np.asarray(input_image,dtype='float64')
    input_image = input_image.copy()
    if input_image.ndim != 2:
        msg = "The data should be a 2D array."
        raise WrongDimensionError(msg)


    # DO THE WAVELET TRANSFORM #############################################

    wavelet_planes_list = []
    wavelet_planes_list.append(input_image)


    for scale_index in range(J):
        previous_scale = wavelet_planes_list[scale_index]

        next_scale = smooth_bspline(previous_scale, 3, scale_index)

        previous_scale -= next_scale

        wavelet_planes_list.append(next_scale)

    coarse = wavelet_planes_list[-1]
    planes = np.zeros((input_image.shape[0],input_image.shape[1],J))

    #Need to do this with a loop as numba doesn't accept np.array(wavelet_planes_list[:-1])
    for i in range(J):
        planes[:,:,i]=wavelet_planes_list[i]

    return coarse, planes  #coarse, all other planes

#####################################
#SUSHI
#Credit: Julia Lascar 2023
#####################################
def SUSHI(X_im,self_T,self_S,Amplitude_S=None,Amp_fixed=False,Amplitude_T=None,
                           niter=10000,stop=1e-6,J=2,kmad=1,mask_amp=10,background=None,
                           Cost_function="Poisson",Chi_2_sigma=None,Custom_Cost=None,
                           Custom_alpha=None):
    """
    Semi-blind Unmixing with Sparsity for Hyperspectral Images

    INPUT:
    X_im: Input hyperspectral image, (l) energy channels, (m,n) pixels
    self_T: IAE model, Thermal
    self_S: IAE model, Synchrotron
    Amp_fixed: if true, Amplitude is fixed to a given value.
    Amplitude_T, Amplitude_S: If Amp_fixed is True, these are the fixed amplitudes
    for the Thermal and Synchrotron models.
    niter: number of maximum iterations
    stop: stopping criterion
    J: number of wavelet scales for spatial regularisation
    kmad: factor for the wavelet sparsity threshold
    mask_amp: pixels with counts under this value will be ignored when
              calculating the regularisation threshold (mad)
    background: option for a background file to be included
    Cost_function: what kind of function is needed. Options are Poisson, Chi_2,
                    and Custom. If Custom is selected, Custom_Cost must be defined.
    Chi_2_sigma: If Chi_2 is selected, this is the standard deviation.
                 The shape of Chi_2_sigma must be the shape of X_im.
                 If none, sigma=1.
    Custom_Cost: Custom cost function. Ignored if Poisson or Chi_2 were selected.
    Custom_alpha: If Custom cost function was selected, please define the amplitude
                  gradient step size. 1/Hessian is recommended.
    ______________________________________________
    OUTPUT:
    Results: A dictionary with the following keys:
        "Params": Dictionary with, for each model, the lambda and amp parameters.
        "XRec": Best fit hyperspectral cube
        "Likelihood": Negative Poisson Log Likelihood over iterations.
    """
    #TO DO:
    #- Make the Parameters only one object instead of being stored in a dict,
    # or list depending on code area. This is a remnant of Jax sometimes
    # raising errors when keeping it all in a dict.
    #
    #- Make this code easily applicable to other unmixing configurations, not
    # hard coded for "Thermal" and "Synchrotron".

    import numpy as onp
    import jax.numpy as np
    from jax import grad, jit, lax, hessian,random,vmap
    from jax.example_libraries.optimizers import adam, momentum, sgd, nesterov, adagrad, rmsprop
    from tqdm.notebook import tqdm,trange
    from astropy.io import fits



    #MODELS AND DIMS
    l,m,n=X_im.shape
    print(f"Shape of the data: {l} channels, {m}x{n} pixels.")
    mask_amp=1 #Ignore wavelet coefficients on pixels with 0 counts.
    X=np.reshape(np.transpose(X_im,(1,2,0)),(m*n,l)) #vectorized image
    if Chi_2_sigma is not None:
        Chi_2_sigma=np.reshape(np.transpose(Chi_2_sigma,(1,2,0)),(m*n,l))
    if background is not None:
        bg_vec=np.reshape(np.transpose(background,(1,2,0)),(m*n,l))
    print("Using SUSHI with background file.")
    print("Using cost function "+Cost_function)

    #Models: the dictionary containing the IAE models.
    Models={"Thermal":{"LearnFunc":self_T},"Synch":{"LearnFunc":self_S}}
    list_models=["Thermal","Synch"]
    for M in Models.keys():
        #number of anchor points:
        N_A=Models[M]["LearnFunc"].AnchorPoints.shape[0]
        Models[M]["N_A"]=N_A

    ################# FUNCTIONS #################
    #def simplex(Lambda,NA):
    #    "Applies simplex constraint"
    #    Norm=onp.tile((Lambda.sum(axis=1)),(NA, 1)).T
    #    Mask=onp.where(Norm!=0)
    #    Lambda[Mask]=Lambda[Mask]/Norm[Mask]
    #    return Lambda,Norm[Mask]
    def X_Recover(Params):
        """
        Spectral Generation function.
        INPUT:
        Params: Dictionary with, for each model, the lambda and amp parameters.
        OUTPUT:
        XRec: Dictionary containg the component spectra and their sum.
        """
        XRec={"Total":np.zeros((m*n,l))}
        for M in list_models:
            LF=Models[M]["LearnFunc"]
            Lambda=Params[M]["Lambda"]
            Amplitude=Params[M]["Amplitude"]
            B = Lambda @ Models[M]["LearnFunc"].PhiE
            XRec[M]=LF.decoder(B)
            XRec[M]=XRec[M]*Amplitude[:,np.newaxis]
            if background is not None:
              XRec["Total"]+=XRec[M]+bg_vec
            else:
              XRec["Total"]+=XRec[M]
        return XRec

    def get_cost(LT,LS,AT,AS,X=X):
        """
        Cost function.
        INPUT:
        LT,LS,AT,AS: Lambda for each models, and amplitude for each models.
        OUTPUT:
        XRec: Negative Poisson Log Likelihood
        """
        Params={"Thermal":{"Lambda":LT,"Amplitude":AT},"Synch":{"Lambda":LS,"Amplitude":AS}}
        XRec=X_Recover(Params)
        Mask=(XRec["Total"]>0)
        if Cost_function=="Poisson":
            Cost=(XRec["Total"]*Mask - X*Mask*np.log(np.abs(XRec["Total"])+1e-16)).sum()
        elif Cost_function=="Chi_2":
            if Chi_2_sigma is not None:
                Cost=(((X*Mask-XRec["Total"]*Mask)**2)/Chi_2_sigma**2).sum()
            else:
                Cost=(((X*Mask-XRec["Total"]*Mask)**2)).sum()
        elif Cost_function=="Custom":
            Cost=Custom_Cost(X,XRec["Total"])
        else:
            print("Error. Cost_function can only be Poisson, Chi_2, or Custom.")
            return np.nan
        #print(Cost)
        return Cost

    def alpha_A(Parameters,m_id,X=X):
        """
        Function to obtain the gradient step for the Amplitude.
        INPUT:
        Parameters: list of parameters [LT,LS,AT,AS]
        m_id: Model id (in order of list_models)
        OUTPUT:
        alpha: Amplitude gradient step.
        """
        #Lambda: Parameters[m_id], Amplitude: Parameters[m_id+2]
        A0,L0=Parameters[m_id+2],Parameters[m_id]
        A1,L1=Parameters[m_id+2+(-1)**m_id],Parameters[np.abs(m_id-1)]
        M0,M1=list_models[m_id],list_models[np.abs(m_id-1)]
        LF0,LF1=Models[M0]["LearnFunc"],Models[M1]["LearnFunc"]
        B0,B1 = L0 @ LF0.PhiE, L1 @ LF1.PhiE
        Phi0,Phi1=LF0.decoder(B0),LF1.decoder(B1)
        if Cost_function == "Poisson":
            H=(Phi0**2)*X/(Phi0*A0[:,onp.newaxis]+Phi1*A1[:,onp.newaxis])**2
            H=H.sum(axis=1)
            alpha=(1/H)
            alpha=alpha.at[onp.where(onp.isnan(alpha))].set(0)
        elif Cost_function == "Chi_2":
            if Chi_2_sigma is not None:
                H=2*Phi0**2/Chi_2_sigma
            else:
                H=2*Phi0**2
            H=H.sum(axis=1)
            alpha=(1/H)
            alpha=alpha.at[onp.where(onp.isnan(alpha))].set(0)
        elif Cost_function == "Custom":
            alpha=Custom_alpha
        return alpha

    #GRADS
    @jit
    def LT_grad(LT,LS,AT,AS,X=X):
        return grad(get_cost,argnums=0)(LT,LS,AT,AS,X=X)
    @jit
    def LS_grad(LT,LS,AT,AS,X=X):
        return grad(get_cost,argnums=1)(LT,LS,AT,AS,X=X)
    @jit
    def AT_grad(LT,LS,AT,AS,X=X):
        return grad(get_cost,argnums=2)(LT,LS,AT,AS,X=X)
    @jit
    def AS_grad(LT,LS,AT,AS,X=X):
        return grad(get_cost,argnums=3)(LT,LS,AT,AS,X=X)
    def mad(z):
        """
        Calculates the median absolute deviation.
        """
        return onp.median(onp.abs(z - onp.median(z)))/0.6735


    def reg_grad_thrs(Lambda,grad,alpha_L,Amp,NA):
        import numpy as onp
        """
        Regularisation for separate Lambdas.

        INPUT:
        Lambda: Lambda map (for one model) (shape: m*n,NA)
        grad: gradient of log-likelihood over lambda (for one model)
        alpha_L: lambda gradient step
        Amp: Amplitude (for one model)
        NA: number of anchor points (for one model)
        OUTPUT:
        Output_vec: Regularised lambda maps (shape: m*n,NA)
        """
        Lambda_map=onp.asarray(onp.reshape(Lambda,(m,n,NA)))
        grad_map=onp.asarray(onp.reshape(grad,(m,n,NA)))
        Output=onp.zeros((m,n,NA))
        Amp=np.reshape(Amp,(m,n))
        for i in range(NA):
            x=Lambda_map[:,:,i]
            c,w = Starlet_Forward2D(x,J=J)
            c_g,w_g = Starlet_Forward2D(grad_map[:,:,i],J=J)
            for r in range(J):
                w_g_mask=w_g[:,:,r].copy()
                w_g_mask=w_g_mask[onp.where(Amp>mask_amp)]
                thrd=kmad*alpha_L*mad(w_g_mask) #sparsity threshold
                #L1 convex relaxation proximal operator
                w[:,:,r] = (w[:,:,r] - thrd*onp.sign(w[:,:,r]))*(onp.abs(w[:,:,r]) > thrd)

            Output[:,:,i]=c + onp.sum(w,axis=2) # Sum all planes including coarse scale
        Output_vec=onp.reshape(Output,(m*n,NA))


        return Output_vec


    Models["Thermal"]["gradL"],Models["Synch"]["gradL"]=LT_grad,LS_grad
    Models["Thermal"]["gradA"],Models["Synch"]["gradA"]=AT_grad,AS_grad


    ################# INIT #################
    Lambda_thermal=np.zeros((m*n,Models["Thermal"]["N_A"])) +1/Models["Thermal"]["N_A"]
    Lambda_synch=np.zeros((m*n,Models["Synch"]["N_A"])) +1/Models["Synch"]["N_A"]

    if Amp_fixed:
        print("Amp fixed")
        Amp_thermal=np.reshape(Amplitude_T,m*n)
        Amp_synch=np.reshape(Amplitude_S,m*n)
    else:
        Amp_thermal=X.sum(axis=1)/2
        Amp_synch=X.sum(axis=1)/2

    #Parameters are stored in a list:
    Parameters=[Lambda_thermal,Lambda_synch,Amp_thermal,Amp_synch]
    Parameters_0=[Lambda_thermal,Lambda_synch,Amp_thermal,Amp_synch]

    #Gradient step size:
    for m_id, M in enumerate(list_models):
        Models[M]["alpha_L"]=0.1/onp.max(X.sum(axis=1))
        Models[M]["alpha_A"]=alpha_A(Parameters,m_id)

    #Values to keep track of
    Acc=[] #Likelihood
    Diff=[] #Difference in likelihood
    LT,LS,AT,AS=Parameters

    ################# LOOP #################
    t=trange(niter, desc='Loss', leave=True)
    for i in t:
        for m_id, M in enumerate(list_models):
            #Descent on Lambda
            LT,LS,AT,AS=Parameters
            Param_old=Parameters[m_id].copy()
            gradL=Models[M]["gradL"]
            Lamb_grad=gradL(LT,LS,AT,AS)
            Lambda_grad_descent=Param_old-Models[M]["alpha_L"]*Lamb_grad

            Lambda_reg=reg_grad_thrs(Lambda_grad_descent,Lamb_grad,
                                     alpha_L=Models[M]["alpha_L"],Amp=Parameters[m_id+2],
                                     NA=Models[M]["N_A"])

            Parameters[m_id]=Lambda_reg

            LT,LS,AT,AS=Parameters

        for m_id, M in enumerate(list_models):
            #Descent on Amplitude
            if not Amp_fixed:
                LT,LS,AT,AS=Parameters
                Param_old=Parameters[m_id+2].copy()
                gradA=Models[M]["gradA"]
                Amp_grad=gradA(LT,LS,AT,AS)
                Models[M]["alpha_A"]=alpha_A(Parameters,m_id)
                new_Amp=Param_old-Models[M]["alpha_A"]*Amp_grad
                new_Amp=new_Amp*(new_Amp>0)
                Parameters[m_id+2]=new_Amp

        LT,LS,AT,AS=Parameters

        likelihood=get_cost(LT,LS,AT,AS)
        Acc.append(likelihood)
        #STOPPING CRITERION
        mean_likelihood=0
        if i>150:
            A1=onp.asarray(Acc[-150:-100])
            A2=onp.asarray(Acc[-50:])
            mean_likelihood=onp.mean(A2-A1)/onp.mean(A1)
            #difference_likelihood=(Acc[i]-Acc[i-30])/Acc[i]
            Diff.append(mean_likelihood)
            #print(mean_likelihood)
            if mean_likelihood<stop:
                print("Stopping criterion reached.")
                break
        t.set_description(f"Loss= {likelihood:.4e}, Mean diff: {mean_likelihood:.2e}")
        t.refresh()

    LT,LS,AT,AS=Parameters
    Results={}
    Params={"Thermal":{"Lambda":LT,"Amplitude":AT},"Synch":{"Lambda":LS,"Amplitude":AS}}
    XRec=X_Recover(Params)
    Results["Params"]=Params
    Results["XRec"]=XRec
    Results["Likelihood"]=Acc

    return Results
