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
def SUSHI(X_im,*models,component_names=["Thermal","Synchrotron"],
          niter=10000,stop=1e-6,J=2,kmad=1,mask_amp=10,background=None,
                           Cost_function="Poisson",Chi_2_sigma=None,Custom_Cost=None,
                           Custom_alpha=None):
    """
    Semi-blind Unmixing with Sparsity for Hyperspectral Images

    INPUT:
    X_im: Input hyperspectral image, (l) energy channels, (m,n) pixels
    models: IAE learnt functions, one for each of the components.
    component_names: names given to the components.
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
    from copy import deepcopy
    import numpy as onp
    import jax.numpy as np
    from jax import grad, jit
    from tqdm.notebook import tqdm,trange

    l,m,n=X_im.shape
    print(f"Shape of the data: {l} channels, {m}x{n} pixels.")
    mask_amp=1 #Ignore wavelet coefficients on pixels with 0 counts.
    X_vec=np.reshape(np.transpose(X_im,(1,2,0)),(m*n,l)) #vectorized image
    if Chi_2_sigma is not None:
        Chi_2_sigma=np.reshape(np.transpose(Chi_2_sigma,(1,2,0)),(m*n,l))
    if background is not None:
        bg_vec=np.reshape(np.transpose(background,(1,2,0)),(m*n,l))
    class Component:
        def __init__(self, name, model):
            self.name = name
            self.model = model
            self.N_A=self.model.AnchorPoints.shape[0]

    Component_list=[]
    print("Models:")
    for i,mod in enumerate(models):
        print(component_names[i])
        Component_list.append(Component(component_names[i], mod))
    N_C=len(Component_list)

    def X_Recover(Theta_all,Amp_all,Comps,X_im=X_im):
        """
        Spectral Generation function.
        """
        l,m,n=X_im.shape
        XRec={"Total":np.zeros((m*n,l))}
        if background is not None:
            XRec["Total"]+=XRec["Total"]+bg_vec

        for index,mod_name in enumerate(component_names):
            #print(mod_name)
            #print(Theta_all[index].shape)
            B = Theta_all[index] @ Comps[index].model.PhiE
            XRec[mod_name]=Comps[index].model.decoder(B)
            XRec[mod_name]=XRec[mod_name]*Amp_all[index][:,np.newaxis]
            XRec["Total"]+=XRec[mod_name]
        return XRec

    def get_cost(Theta_all,Amp_all,Comps):
        XRec=X_Recover(Theta_all,Amp_all,Comps)
        Mask=(XRec["Total"]>0)
        if Cost_function=="Poisson":
            Cost=(XRec["Total"]*Mask - X_vec*Mask*np.log(np.abs(XRec["Total"])+1e-16)).sum()
        elif Cost_function=="Chi_2":
            if Chi_2_sigma is not None:
                Cost=(((X_vec*Mask-XRec["Total"]*Mask)**2)/Chi_2_sigma**2).sum()
            else:
                Cost=(((X_vec*Mask-XRec["Total"]*Mask)**2)).sum()
        elif Cost_function=="Custom":
            Cost=Custom_Cost(X_vec,XRec["Total"])
        else:
            print("Error. Cost_function can only be Poisson, Chi_2, or Custom.")
            return np.nan
        return Cost

    def get_cost_with_varyingAT(T,A,Theta_all,Amp_all,index,Comps):
        #CHECK HERE: used to be Amp_all_var !!!
        #In a separate function to calculate the
        Amp_all_var=deepcopy(Amp_all)
        Amp_all_var[index]=A

        Theta_all_var=deepcopy(Theta_all)
        Theta_all_var[index]=T
        #print("varying T",T.shape,"index:", index)
        cost=get_cost(Theta_all_var,Amp_all_var,Comps)
        return cost

    for fixed_index in range(N_C):
        #Getting the gradients for each components
        #print("index",fixed_index)
        Component_list[fixed_index].T_grad= jit(lambda T,A,T_all,A_all,fixed_index:
                                                grad(get_cost_with_varyingAT,argnums=0)(
                                                    T,A,T_all,A_all,fixed_index,Component_list),
                                               static_argnames="fixed_index")
        Component_list[fixed_index].A_grad= jit(lambda T,A,T_all,A_all,fixed_index:
                                                grad(get_cost_with_varyingAT,argnums=1)(
                                                    T,A,T_all,A_all,fixed_index,Component_list),
                                               static_argnames="fixed_index")

        #GradT=grad(get_cost_with_varyingAT,argnums=0)
        #globals()[f"GradT_{fixed_index}"]=deepcopy(GradT)
        #GradA=grad(get_cost_with_varyingAT,argnums=1)
        #globals()[f"GradA_{fixed_index}"]=deepcopy(GradA)
        #Component_list[fixed_index].T_grad= jit(lambda T,A,T_all,A_all: globals()[f"GradT_{fixed_index}"](T,A,T_all,A_all,fixed_index,Component_list))
        #Component_list[fixed_index].A_grad= jit(lambda T,A,T_all,A_all: globals()[f"GradA_{fixed_index}"](T,A,T_all,A_all,fixed_index,Component_list))



    def mad(z):
        """
        Calculates the median absolute deviation.
        """
        return onp.median(onp.abs(z - onp.median(z)))/0.6735


    def reg_grad_thrs(Lambda,grad,alpha_T,Amp,NA):
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
        import numpy as onp
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
                thrd=kmad*alpha_T*mad(w_g_mask) #sparsity threshold
                #L1 convex relaxation proximal operator
                w[:,:,r] = (w[:,:,r] - thrd*onp.sign(w[:,:,r]))*(onp.abs(w[:,:,r]) > thrd)

            Output[:,:,i]=c + onp.sum(w,axis=2) # Sum all planes including coarse scale
        Output_vec=onp.reshape(Output,(m*n,NA))
        return Output_vec


    def alpha_A_generator(Comp,all_theta,all_amp,i,X=X_vec):
        """
        Function to obtain the gradient step for the Amplitude.
        INPUT:
        Parameters: list of parameters [LT,LS,AT,AS]
        m_id: Model id (in order of list_models)
        OUTPUT:
        alpha: Amplitude gradient step.
        """
        #Lambda: Parameters[m_id], Amplitude: Parameters[m_id+2]

        A0=all_amp[i]
        Phi0=Comp[i].model.decoder(all_theta[i]@Comp[i].model.PhiE)
        if Cost_function == "Poisson":
            dividend=Phi0*A0[:,onp.newaxis]
            k=0
            for j in range(N_C):
                if j!=i:
                    Phi_other_J= Comp[j].model.decoder(all_theta[j]@Comp[j].model.PhiE)
                    dividend+=Phi_other_J*all_amp[j][:,onp.newaxis]
                    k+=1
            H=(Phi0**2)*X/(dividend)**2
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

    Amp_all=[]
    Theta_all=[]
    for c in range(N_C):
        #First guess
        Amp_all.append((X_vec.sum(axis=1))/N_C)
        Theta_all.append(np.ones((m*n,Component_list[c].N_A))/Component_list[c].N_A)

    #Gradient step size
    alpha_T=0.1/onp.max(X_vec.sum(axis=1))

    #Values to keep track of
    Acc=[] #Likelihood
    Diff=[] #Difference in likelihood

    ################# LOOP #################
    t=trange(niter, desc='Loss', leave=True)
    for i in t:
        for c in range(N_C):
            #Descent on Lambda
            #print("Theta before",Theta_all[c].shape)
            #print("Theta_all",Theta_all[0].shape,Theta_all[1].shape)
            gradT=Component_list[c].T_grad
            #print("grad_T",gradT)
            Theta_grad=gradT(Theta_all[c],Amp_all[c],Theta_all,Amp_all,fixed_index=c)
            Theta_grad_descent=Theta_all[c]-alpha_T*Theta_grad

            Theta_reg=reg_grad_thrs(Theta_grad_descent,Theta_grad,
                                     alpha_T=alpha_T,Amp=Amp_all[c],
                                     NA=Component_list[c].N_A)

            Theta_all[c]=Theta_reg
            #print("Theta_all",Theta_all[0].shape,Theta_all[1].shape)
            #print("Theta after",Theta_reg.shape)



        for c in range(N_C):
            #Descent on Amplitude
            gradA=Component_list[c].A_grad
            Amp_grad=gradA(Theta_all[c],Amp_all[c],Theta_all,Amp_all,fixed_index=c)
            alpha_A=alpha_A_generator(Component_list,Theta_all,Amp_all,c)
            new_Amp=Amp_all[c]-alpha_A*Amp_grad
            new_Amp=new_Amp*(new_Amp>0) #non-negative
            Amp_all[c]=new_Amp

        likelihood=get_cost(Theta_all,Amp_all,Component_list)
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

    Results={}
    XRec=X_Recover(Theta_all,Amp_all,Component_list)
    Results["Theta"]=Theta_all
    Results["Amplitude"]=Amp_all
    #Results["Components"]=Component_list
    Results["XRec"]=XRec
    Results["Likelihood"]=Acc

    return Results
