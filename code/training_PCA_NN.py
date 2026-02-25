from jax.example_libraries import stax
from jax.example_libraries import optimizers
from jax import random as jran
from jax import jit
from jax import value_and_grad
import jax.numpy as jnp
import numpy as np
import pickle
from time import time
from tqdm import trange
from sklearn.model_selection import train_test_split

# HAVE FUNCTIONS TO 
#TRAIN
#TEST
#AND LOAD



#################################################


# DATA TRANSFORMATIONS
def transform_spec(spec,constants=None,rank=300):
	"""
	A function that takes data from the spectral domain, applies a normalization, 
	a log function, a PCA, and finally, a rescaling. 

	INPUT
	spec: the data to be transformed. Needs to be of shape n_channels x n_entries. 
	OUTPUT
	transformed_spec: the transformed data. 
	constants: list of constants to be kept to do the inverse transformation.

	"""
	if constants is None:
		spec=np.log(spec/(spec.sum(0)[np.newaxis,:]))
		spec_log_mean,spec_log_std=spec.mean(1),spec.std(1)
		spec_log_scaled = (spec-spec_log_mean[:,np.newaxis])/spec_log_std[:,np.newaxis]
		Cov=(spec_log_scaled)@(spec_log_scaled).T
		S_spec,U_spec=np.linalg.eig(Cov)
		U_spec=U_spec[:,:rank]
		spec_PCA=spec_log_scaled.T@U_spec
		spec_PCA_mean,spec_PCA_std=spec_PCA.mean(),spec_PCA.std()
		transformed_spec=(spec_PCA-spec_PCA_mean)/spec_PCA_std
		constants=[spec_log_mean,spec_log_std,U_spec,spec_PCA_mean,spec_PCA_std]
		return transformed_spec,constants
	else:
		spec_log_mean,spec_log_std,U_spec,spec_PCA_mean,spec_PCA_std=constants
		spec=np.log(spec/(spec.sum(0)[np.newaxis,:]))
		spec_log_scaled = (spec-spec_log_mean[:,np.newaxis])/spec_log_std[:,np.newaxis]
		spec_PCA=spec_log_scaled.T@U_spec
		transformed_spec=(spec_PCA-spec_PCA_mean)/spec_PCA_std
		return transformed_spec


def untransform_spec(spec_PCA,constants):
	"""
	The reverse transformation of transform_spec. 
	"""
	spec_log_mean,spec_log_std,U_spec,spec_PCA_mean,spec_PCA_std=constants
	spec= (np.e**(((spec_PCA*spec_PCA_std+spec_PCA_mean
                           )@U_spec.T)*spec_log_std
                          +spec_log_mean))
	return spec 

def transform_physpar(phys_params,constants=None):
	"""
	A function that takes physical parameters and standardize them.

	INPUT
	spec: the physical parameters to be transformed. Needs to be of shape n_entries x n_phys. 
	OUTPUT
	transformed_params: the transformed physical parameters. 
	constants: list of constants to be kept to do the inverse transformation.
	"""
	if constants is None:
		phys_params_mean=phys_params.mean(0)
		phys_params_std=phys_params.std(0)
		transformed_params=(phys_params-phys_params_mean[np.newaxis,:])/phys_params_std[np.newaxis,:]
		constants=[phys_params_mean,phys_params_std]
		return transformed_params,constants
	else:
		phys_params_mean,phys_params_std=constants
		transformed_params=(phys_params-phys_params_mean[np.newaxis,:])/phys_params_std[np.newaxis,:]
		return transformed_params

def untransform_physpar(params_trans, constants):
	"""
	The reverse transformation of transform_physpar. 
	"""
	phys_params_mean,phys_params_std=constants
	phys_params=params_trans*phys_params_std[np.newaxis,:]+phys_params_mean[np.newaxis,:]
	return phys_params

#################################################

#SPLITTING DATA

def transform_and_split_data(phys_params, spec, rank=300,random_state=123,train_size=0.8):

	spec_PCA,spec_cnst=transform_spec(spec,rank=rank)
	params_trans,params_cnst=transform_physpar(phys_params)
	#Split data
	
	X_train, X_test, Y_train, Y_test = train_test_split(params_trans, spec_PCA, 
		train_size=train_size,random_state=random_state)
	X_train, X_test, Y_train, Y_test = jnp.array(X_train),\
	                                   jnp.array(X_test),\
	                                   jnp.array(Y_train),\
	                                   jnp.array(Y_test)

	return ({"X_train":X_train, "X_test":X_test, 
			'Y_train':Y_train, 'Y_test':Y_test,
			'spec_cnst':spec_cnst,'params_cnst':params_cnst})

def check_rank(spec,rank):
	spec_PCA,cnst=transform_spec(spec,rank=rank)
	#print(cnst)
	spec_return_from_PCA=untransform_spec(spec_PCA,cnst).T
	MSE=((spec-spec_return_from_PCA)**2).mean()
	return spec_PCA,spec_return_from_PCA,MSE


#TRAINING A MODEL 



def train_model(phys_params, spec,rank=300,
	seed_train=42,seed_split=123,split=0.8,vocal=True,
	batch_size = 500,num_batches = 5e4,save=True,name="PCA_NN"):
	
	if vocal:
		print("Transforming data.")

	train_direc=transform_and_split_data(phys_params, spec, 
		rank=rank,random_state=seed_split,train_size=split)
	if vocal:
		print("Training the network.")

	n_features=train_direc["X_train"].shape[1]
	n_targets=train_direc["Y_train"].shape[1]
	ran_key = jran.PRNGKey(seed_train)
	ran_key, net_init_key = jran.split(ran_key)
	

	net_init, net_apply = stax.serial(
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(n_targets),
	)
	out_shape, net_params = net_init(net_init_key, input_shape=(-1, n_features))
	opt_init, opt_update, get_params = optimizers.adam(1e-4)
	opt_state = opt_init(net_params)

	Weights_PCA=1
	@jit
	def weighted_mse_loss(params, loss_data):
	    X_tbatch, targets = loss_data
	    preds = net_apply(params, X_tbatch)
	    diff = preds - targets 
	    return jnp.sum(Weights_PCA*diff * diff) #not weighted: jnp.mean(diff * diff)

	@jit
	def train_step(step_i, opt_state, loss_data):
	    net_params = get_params(opt_state)
	    loss, grads = value_and_grad(weighted_mse_loss, argnums=0)(net_params, loss_data)
	    return loss, opt_update(step_i, grads, opt_state)

	if vocal:
		print('First layer shapes: ', np.shape(net_params[0][0]), np.shape(net_params[0][1]))
		print('Final layer shapes: ', np.shape(net_params[-1][0]), np.shape(net_params[-1][1]))

	loss_history = []
	start = time()
	if vocal: 
		t=trange(num_batches,desc='Loss', leave=True)
	else:
		t=range(num_batches)

	for ibatch in t:
		ran_key, batch_key = jran.split(ran_key)
		indices=jran.randint(batch_key,shape=batch_size,minval=0,maxval=train_direc["X_train"].shape[0])
		X_batch=train_direc["X_train"][indices,:]
		Y_batch=train_direc["Y_train"][indices,:]
		loss_data = X_batch, Y_batch
		loss, opt_state = train_step(ibatch, opt_state, loss_data)
		loss_history.append(float(loss))
		if vocal:
			if ibatch%100==0:
				t.set_description(f"Loss= {loss:.4e}, started with: {loss_history[0]:.4e}")
				t.refresh()

	end = time()
	msg = "training time for {0} iterations = {1:.1f} seconds"
	if vocal:
		print(msg.format(num_batches, end-start))
	if save:
		file_name=f'{name}_batches_{num_batches}_seed_{seed_train}.p'
		trained_params = optimizers.unpack_optimizer_state(opt_state)
		saved={"loss":loss,"trained_params":trained_params,"train_direc":train_direc}
		with open(file_name,"wb") as f:
		    pickle.dump(saved, f)
	return {"loss":loss_history,"opt_state":opt_state,"train_direc":train_direc}

#################################################

#LOADING A MODEL


#This should be usable without any other function or variable 
def load_model(model_path): 
	"""
	param_path: location of the "opt_state" weights.
	transform_path: location of the directory that contains all the necessary information 
	to transform from the physical to the latent space. 
	"""
	with open(model_path,"rb") as f:
		model_dic = pickle.load(f)
		best_opt_state= optimizers.pack_optimizer_state(model_dic["trained_params"])
		train_direc=model_dic["train_direc"]
	

	n_features=train_direc["X_train"].shape[1]
	n_targets=train_direc["Y_train"].shape[1]
	_, _, get_params = optimizers.adam(1e-4)
	net_init, net_apply = stax.serial(
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(256), stax.Gelu,
	    stax.Dense(n_targets),
	)
	params_cnst=train_direc["params_cnst"]
	spec_cnst=train_direc["spec_cnst"]
	def return_to_spectra(spec_PCA):
		return untransform_spec(spec_PCA,constants=spec_cnst)
	def scale_phys_params(phys_params):
		return transform_physpar(phys_params,constants=params_cnst)

	def neural_network_from_phys(phys_params):
		return return_to_spectra(
			net_apply(
				get_params(best_opt_state),
				scale_phys_params(phys_params)))
	def neural_network(trans_param):
		return net_apply(get_params(best_opt_state),trans_param)
	return neural_network,neural_network_from_phys,params_cnst,spec_cnst




