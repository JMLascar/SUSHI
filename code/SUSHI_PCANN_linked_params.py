# Functions

from copy import deepcopy as dp
import scipy.linalg as lng
import copy as cp
import warnings
from jax import value_and_grad

from jax import jit
import scipy as scp
import pickle
import numpy as np 
from matplotlib import pyplot as plt
import sys, os

sys.path.insert(0,'code/')
from training_PCA_NN import untransform_physpar
from pyStarlet_2D1D_jax import *
import warnings
from ipywidgets import interact, fixed
from tqdm.notebook import tqdm,trange
from copy import deepcopy
import numpy as onp    
import jax.numpy as np
from jax import grad, jit

#####################################
#SUSHI
#Credit: Julia Lascar 2023 / Jax acceleration 2024 / Linked parameters: 2025
#####################################
def SUSHI(X_im,*models_dir,component_names=["Thermal","Synchrotron"],
          niter=10000,stop=1e-6,J=2,kmad=1,background=None,
                           Cost_function="Poisson",
                           Chi_2_sigma=None,Custom_Cost=None,
                           alpha_A=None,alpha_T=None,
                           intermediate_save=False,
                           file_name="Sushi_result",
                           save=100,restart=None):
    """
    Semi-blind Unmixing with Sparsity for Hyperspectral Images

    INPUT:
    X_im: Input hyperspectral image, (l) energy channels, (m,n) pixels
    models_dir: dictionaries of the spectral models, one for each of the components.
                These should have the following keys: 
                "model": the PCA-NN network for this model.
                "param_names": list of names for the physical parameters (kT,z,nH...). 
                    IMPORTANT: Linked parameters should have the same name. If you have models with similar 
                    parameters that should NOT be linked, please give them different names (e.g. kT_1 and kT_2).
                "first_param": list of first guess parmameters (in normalized units, not physical).
                "limits:" list of min and max values for the parameters: [[min_1,max_1],[min_2,max_2],...]
    component_names: List of names given to the models. These should be in the same order as the models. 
                    ["Model1","Model2",...]
    niter: number of maximum iterations
    stop: stopping criterion. Set to np.nan to ignore.
    J: number of wavelet scales for spatial regularisation
    kmad: factor for the wavelet sparsity threshold
    background: option for a background array to be included. It can either be a spectra (size: l), 
                or a cube (l,m,n).
    Cost_function: what kind of function is needed. Options are Poisson, Chi_2,
                    and Custom. If Custom is selected, Custom_Cost must be defined.
    Chi_2_sigma: If Chi_2 is selected, this is the standard deviation.
                 The shape of Chi_2_sigma must be the shape of X_im.
                 If none, sigma=1.
    Custom_Cost: Custom cost function. Ignored if Poisson or Chi_2 were selected.
    Custom_alpha: If Custom cost function was selected, please define the amplitude
                  gradient step size. 1/Hessian is recommended.
    alpha_T: value for the gradient step on theta. Set to None for automatic calculation.
    intermediate_save: if True, the result will be saved every N iterations. 
    save: the number of iterations at which the result is saved. 
    restart: Option to set a past output of SUSHI to make the algorithm calculate more iterations.
             Set this to the output of the SUSHI algorithm, and it will pick up where it left off. 
             If None, this is ignored. 

    ______________________________________________
    OUTPUT:
    Results: A dictionary with the following keys:
        "Theta": The best fit spectral parameters. Size: m,n,total number of parameters.
        Order of paramereters is Model 1: p1, p2,..., Model 2: p1, p2 ,..., and so on.
        "Amplitude": The brightness map of each component. Size: m,n,number of components.
        "XRec": Dictionary of the best fit hyperspectral cube. Each component is of size (l,m,n).
                The "Total" key is the sum of the individual components.
        "Likelihood": Negative Poisson Log Likelihood over iterations.
    """


    ########################## SET UP ##########################
    l,m,n=X_im.shape
    L,M,N=l,m,n
    print(f"Shape of the data: {l} channels, {m}x{n} pixels.")
    mask_amp=1 #Ignore wavelet coefficients on pixels with 0 counts.
    X_vec=np.reshape(np.transpose(X_im,(1,2,0)),(m*n,l)) #vectorized image
    class Component:
        def __init__(self, name, model,param_names,first_param,params_cnst):
            self.name = name
            self.model = model
            self.N_P=len(param_names) ##to change
            self.first_param=first_param
            self.param_names=param_names
            self.params_cnst=params_cnst

    Component_list=[]
    print("Models:")
    for i,mod in enumerate(models_dir):
        print(component_names[i])
        Component_list.append(Component(component_names[i], mod["model"],
            mod["param_names"],mod["first_param"],mod["params_cnst"]))
    N_C=len(Component_list)
    print(f"Number of components: {N_C}")

    ### Determining if any parameter should be linked
    Param_names=[]
    for c in range(N_C):
        Param_names.append(Component_list[c].param_names)

    if N_C==2:
        Sets=[set(Param_names[i]) for i in range(len(Param_names))]
        linked_params=set.intersection(*Sets)

    elif N_C>2:
        linked_params=set()
        for i in range(N_C-1):
            Sets=[set(Param_names[i]),set(Param_names[i+1])]
            linked_params=linked_params.union(set.intersection(*Sets))
    Link_Count=onp.zeros(len(linked_params))
    Parent_Model_List=onp.zeros(len(linked_params),dtype=int)
    Parent_Param_List=onp.zeros(len(linked_params),dtype=int)
    for c in range(N_C):
        Component_list[c].linked=[]
        for i,p in enumerate(Param_names[c]):
            for j,s in enumerate(linked_params):
                if bool(set([s]) & set([p])):
                    if Link_Count[j]==0:
                        Parent_Model_List[j]=c
                        Parent_Param_List[j]=i
                        Link_Count[j]=1
                        Component_list[c].linked.append({"param_name":p,"param_index":i,
                                "parent_model_index":Parent_Model_List[j],
                                     "parent_param_index":Parent_Param_List[j]})
                    else:
                        Component_list[c].linked.append({"param_name":p,"param_index":i,
                                "parent_model_index":Parent_Model_List[j],
                                     "parent_param_index":Parent_Param_List[j]})


    def Link_Params(Theta):
        for index in range(N_C):
            Comp=Component_list[index]
            if len(Comp.linked)>0:
                for s in Comp.linked:
                    if s["parent_model_index"]!=index:
                        Theta_c=theta_c(Theta,s["parent_model_index"])
                        NewTheta=Theta_c[:,s["parent_param_index"]]
                        Theta=Theta.at[:,NP_arraystart[c]+s["param_index"]].set(NewTheta)
        return Theta

    for index in range(N_C):
            Comp=Component_list[index]
            if len(Comp.linked)>0:
                for s in Comp.linked:
                    if s["parent_model_index"]!=index:
                        Child_Model=component_names[index]
                        Child_Parameter=Comp.param_names[s["param_index"]]
                        Parent_Model=component_names[s['parent_model_index']]
                        Parent_Parameter=Component_list[s['parent_model_index']].param_names[s['parent_param_index']]

                        print(f"Parameter  {Child_Parameter} in model {Child_Model}",
                            f" is linked to Parameter {Parent_Parameter}",
                            f"in model {Parent_Model}")

                    else:
                        print(f"Parameter {Comp.param_names[s['param_index']]} marked as linked.")
                        print(f"Model {component_names[index]} marked as parent model.")


    NP_total=0
    NP_array=onp.zeros(N_C,dtype=int)
    NP_arraystart=onp.zeros(N_C,dtype=int)
    for c in range(N_C):
        NP_total+=Component_list[c].N_P
        NP_array[c]=int((Component_list[c].N_P))
    for c in range(N_C-1):
        NP_arraystart[c+1]=NP_arraystart[c]+int(Component_list[c].N_P)

    if background is not None:
        if len(background.shape)==1:
            bg_vec=np.zeros((m*n,l))
            for i in range(l):
                bg_vec=bg_vec.at[:,i].set(background[i])
        else:
            bg_vec=np.reshape(background,(l,m*n)).T
    else:
        bg_vec=0

    def theta_c(Theta_all,c):
        return Theta_all[:,NP_arraystart[c]:NP_arraystart[c]+NP_array[c]]
    def X_Recover(Theta_all,Amp_all,Comps,X_im=X_im):
        """
        Spectral Generation function with linked params.
        """
        l,m,n=X_im.shape
        XRec={"Total":np.zeros((m*n,l))}
        XRec["Total"]+=XRec["Total"]+bg_vec
        Theta_all=Link_Params(Theta_all)
        for index,mod_name in enumerate(component_names):
            Theta=theta_c(Theta_all,index)
            XRec[mod_name]=Comps[index].model(Theta)
            XRec[mod_name]=jnp.real(XRec[mod_name]*Amp_all[index,:,np.newaxis])
            XRec["Total"]+=XRec[mod_name]
        return XRec
    
    def get_cost(Theta_all,Amp_all,Comps):
        """
        Cost function.
        """
        XRec=X_Recover(Theta_all,Amp_all,Comps)
        Mask=(XRec["Total"]>0)
        if Cost_function=="Poisson":
            Cost=(XRec["Total"]*Mask - X_vec*Mask*np.log(np.abs(XRec["Total"])+1e-14)).sum()
        elif Cost_function=="Chi_2":
            if Chi_2_sigma is not None:
                Cost=(((X_vec*Mask-XRec["Total"]*Mask)**2)/Chi_2_sigma**2).sum()
            else:
                Cost=(((X_vec*Mask-XRec["Total"]*Mask)**2)).sum()
        elif Cost_function=="Custom":
            Cost=Custom_Cost(X_vec,XRec["Total"])
        else:
            prin("Error. Cost_function can only be Poisson, Chi_2, or Custom.")
            return np.nan
        return Cost

    for fixed_index in range(N_C):
        #Getting the gradients for each components
        T_grad= jit(lambda T,A: grad(get_cost,argnums=0)(T,A,Component_list))
        A_grad= jit(lambda T,A: grad(get_cost,argnums=1)(T,A,Component_list))


    def mad(z):
        """
        Calculates the median absolute deviation.
        """
        return onp.median(onp.abs(z - onp.median(z)))/0.6735


    def reg_grad_thrs(Lambda,grad,alpha_T,Amp,NA):
        """
        Regularisation for separate maps of parameters.

        INPUT:
        Lambda: Parameter map (for one model) (shape: m*n,NA)
        grad: gradient of log-likelihood over lambda (for one model)
        alpha_L: lambda gradient step
        Amp: Amplitude (for one model)
        NA: number of anchor points (for one model)
        OUTPUT:
        Output_vec: Regularised lambda maps (shape: m*n,NA)
        """
        Lambda_map=jnp.asarray(jnp.reshape(Lambda,(M,N,NA)))
        grad_map=jnp.asarray(jnp.reshape(grad,(M,N,NA)))
        Output=jnp.zeros((M,N,NA))
        Amp=jnp.reshape(Amp,(M,N))
        for i in range(NA):
            x=Lambda_map[:,:,i]
            c,w = Starlet_Forward2D(x,J=J,M=M,N=N)
            c_g,w_g = Starlet_Forward2D(grad_map[:,:,i],J=J,M=M,N=N)
            for r in range(J):
                w_g_mask=w_g[r,:,:].copy()
                w_g_mask=w_g_mask[jnp.where(Amp>mask_amp)]
                thrd=kmad*alpha_T*mad(w_g_mask) #sparsity threshold
                #L1 convex relaxation proximal operator
                w=w.at[r,:,:].set((w[r,:,:] - thrd*jnp.sign(w[r,:,:]))*(jnp.abs(w[r,:,:]) > thrd))
            Output=Output.at[:,:,i].set(c + jnp.sum(w,axis=0))# Sum all planes including coarse scale
        Output_vec=jnp.reshape(Output,(M*N,NA))
        return Output_vec

    #First guess
    Amp_all=[]
    Theta_all=np.zeros((m*n,NP_total))
    for c in range(N_C):
        Amp_all.append((X_vec.sum(axis=1))/N_C)
        for i in range(Component_list[c].N_P):
            Theta_all=Theta_all.at[:,NP_arraystart[c]+i].set(Component_list[c].first_param[i])
    Amp_all=onp.reshape(Amp_all,(N_C,m*n))
    
    #Gradient step size
    if alpha_T is None:
        alpha_T=0.1/onp.max(X_vec.sum(axis=1))
    if alpha_A is None:
        alpha_A=1

    #Values to keep track of
    Acc=[] #Likelihood
    Diff=[] #Difference in likelihood

    #If starting  SUSHI again from obtained results (more iterations)
    if restart is not None:
        Theta_all=np.reshape(restart["Theta"],(m*n,NP_total))
        Amp_all=np.reshape(restart["Amplitude"],(m*N,N_C)).T
        XRec=restart["XRec"]
        Acc=restart["Likelihood"]

    ################# LOOP #################
    t=trange(niter, desc='Loss', leave=True)
    for i in t:
        #Descent on Theta
        Theta_grad=T_grad(Theta_all,Amp_all)
        Theta_grad_descent=Theta_all-alpha_T*Theta_grad
        for c in range(N_C):
            Theta_gd=Theta_grad_descent[:,NP_arraystart[c]:NP_arraystart[c]+NP_array[c]]
            Theta_g=Theta_grad_descent[:,NP_arraystart[c]:NP_arraystart[c]+NP_array[c]]
            #Regularizing Theta
            Theta_reg=reg_grad_thrs(Theta_gd,Theta_g,
                                     alpha_T=alpha_T,Amp=Amp_all[c,:],
                                     NA=Component_list[c].N_P)
            
            Theta_all=Theta_all.at[:,NP_arraystart[c]:NP_arraystart[c]+NP_array[c]].set(Theta_reg)
        Theta_all=Link_Params(Theta_all)

        
        #Descent on Amplitude
        Amp_grad=A_grad(Theta_all,Amp_all)
        new_Amp=Amp_all-alpha_A*Amp_grad
        #Ensure non-negative Amplitude
        new_Amp=new_Amp*(new_Amp>0) 
        Amp_all=new_Amp
        Amp_all=Amp_all.at[np.where(Amp_all==0)].set(1e-14)
        #Calculating the cost
        likelihood=get_cost(Theta_all,Amp_all,Component_list)
        Acc.append(likelihood)
        #Stopping criterion
        mean_likelihood=0
        if i>150:
            A1=onp.asarray(Acc[-150:-100])
            A2=onp.asarray(Acc[-50:])
            mean_likelihood=onp.mean(A2-A1)/onp.mean(A1)
            Diff.append(mean_likelihood)
            if mean_likelihood<stop:
                print("Stopping criterion reached.")
                break
        t.set_description(f"Loss= {likelihood:.4e}, Mean diff: {mean_likelihood:.2e}")
        t.refresh()
        if  intermediate_save:
            if i%save==0:
                Results={}
                XRec=X_Recover(Theta_all,Amp_all,Component_list)
                Results["Theta"]=Theta_all
                Results["Amplitude"]=Amp_all
                #Results["Components"]=Component_list
                Results["XRec"]=XRec
                Results["Likelihood"]=Acc
                with open(f"{file_name}_{i}.p","wb") as f:
                    pickle.dump(Results,f)
    Results={}
    XRec=X_Recover(Theta_all,Amp_all,Component_list)
    Theta_res=[]
    for c in range(N_C):
        T=untransform_physpar(theta_c(Theta_all,c),(Component_list[c].params_cnst))
        Theta_res.append(np.reshape(T,(m,n,Component_list[c].N_P)))
    Results["Theta"]=Theta_res

    Results["Amplitude"]=np.reshape(Amp_all.T,(m,n,N_C))
    #Results["Components"]=Component_list
    Results["XRec"]=XRec
    Results["XRec"]["Total"]=np.reshape(Results["XRec"]["Total"].T,(l,m,n))
    for index,mod_name in enumerate(component_names):
        Comp=Component_list[index]
        Results["XRec"][mod_name]=np.reshape(Results["XRec"][mod_name].T,(l,m,n))
    Results["Likelihood"]=Acc
    if np.isnan(Acc[-1]):
        print("NaN values in the likelihood.")
        print("This problem can usually be fixed by lowering the gradient step (alpha_T,alpha_A).")
    with open(f"{file_name}.p","wb") as f:
        pickle.dump(Results,f)
    return Results

