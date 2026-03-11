import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
from time import time
from tqdm import trange
from sklearn.model_selection import train_test_split

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
		U_spec=np.real(U_spec[:,:rank])
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
	U_spec = np.real(U_spec)
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
	
	X_train, X_test, Y_train, Y_test = np.array(X_train),\
	                                   np.array(X_test),\
	                                   np.array(Y_train),\
	                                   np.array(Y_test)

	return ({"X_train":X_train, "X_test":X_test, 
			'Y_train':Y_train, 'Y_test':Y_test,
			'spec_cnst':spec_cnst,'params_cnst':params_cnst})

def check_rank(spec,rank):
	spec_PCA,cnst=transform_spec(spec,rank=rank)
	spec_return_from_PCA=untransform_spec(spec_PCA,cnst).T
	MSE=((spec-spec_return_from_PCA)**2).mean()
	return spec_PCA,spec_return_from_PCA,MSE


#TRAINING A MODEL 

def build_net(n_features, n_targets):
	return nn.Sequential(
		nn.Linear(n_features, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, 256), nn.GELU(),
		nn.Linear(256, n_targets)
	)

def train_model(phys_params, spec,rank=300,
	seed_train=42,seed_split=123,split=0.8,vocal=True,
	batch_size = 500,num_batches = 5e4,save=True,name="PCA_NN"):
	
	num_batches = int(num_batches)

	if vocal:
		print("Transforming data.")

	train_direc=transform_and_split_data(phys_params, spec, 
		rank=rank,random_state=seed_split,train_size=split)
	if vocal:
		print("Training the network.")

	device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
	if vocal:
		print(f"Using device: {device}")

	torch.manual_seed(seed_train)

	n_features=train_direc["X_train"].shape[1]
	n_targets=train_direc["Y_train"].shape[1]
	
	model = build_net(n_features, n_targets).to(device)
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	
	# Prepare dataloader for infinite random sampling like the original code
	X_train_t = torch.tensor(train_direc["X_train"], dtype=torch.float32)
	Y_train_t = torch.tensor(train_direc["Y_train"], dtype=torch.float32)
	
	dataset = TensorDataset(X_train_t, Y_train_t)
	# Infinite dataloader pattern or random choice
	
	Weights_PCA = 1.0

	loss_history = []
	start = time()
	if vocal: 
		t=trange(num_batches,desc='Loss', leave=True)
	else:
		t=range(num_batches)

	model.train()
	
	# PyTorch is eager so we just sample manually as in the JAX code
	# Alternatively, if num_batches represents epochs, we'd use Dataloader. 
	# The original did num_batches random slices explicitly.
	
	indices_generator = torch.randint(0, len(X_train_t), (num_batches, batch_size))

	for ibatch in t:
		idx = indices_generator[ibatch]
		X_batch = X_train_t[idx].to(device)
		Y_batch = Y_train_t[idx].to(device)
		
		optimizer.zero_grad()
		preds = model(X_batch)
		diff = preds - Y_batch
		loss = torch.sum(Weights_PCA * diff * diff)
		
		loss.backward()
		optimizer.step()
		
		loss_val = loss.item()
		loss_history.append(loss_val)
		
		if vocal:
			if ibatch%100==0:
				t.set_description(f"Loss= {loss_val:.4e}, started with: {loss_history[0]:.4e}")
				t.refresh()

	end = time()
	msg = "training time for {0} iterations = {1:.1f} seconds"
	if vocal:
		print(msg.format(num_batches, end-start))
	if save:
		file_name=f'{name}_batches_{num_batches}_seed_{seed_train}.p'
		# Only save the state dict instead of full model or raw weights
		saved={
			"loss":loss_history,
			"model_state_dict":model.state_dict(),
			"train_direc":train_direc,
		}
		with open(file_name,"wb") as f:
		    pickle.dump(saved, f)
	
	return {"loss":loss_history,"model":model,"train_direc":train_direc}

#################################################

#LOADING A MODEL

def load_model(model_path): 
	"""
	model_path: location of the trained model pickle file.
	"""
	with open(model_path,"rb") as f:
		model_dic = pickle.load(f)
		train_direc=model_dic["train_direc"]
	
	n_features=train_direc["X_train"].shape[1]
	n_targets=train_direc["Y_train"].shape[1]
	
	device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

	model = build_net(n_features, n_targets)
	
	if "model_state_dict" in model_dic:
		model.load_state_dict(model_dic["model_state_dict"])
	elif "trained_params" in model_dic:
		raise ValueError("This model was trained with JAX and cannot be loaded directly via PyTorch without manual weight conversion.")
	else:
		raise ValueError("Invalid PyTorch model file.")

	model.eval()
	model.to(device)

	params_cnst=train_direc["params_cnst"]
	spec_cnst=train_direc["spec_cnst"]
	
	spec_cnst = [np.real(c) for c in spec_cnst]
	params_cnst = [np.real(c) for c in params_cnst]
	
	def return_to_spectra(spec_PCA):
		return untransform_spec(spec_PCA,constants=spec_cnst)
	def scale_phys_params(phys_params):
		return transform_physpar(phys_params,constants=params_cnst)

	def neural_network_from_phys(phys_params):
		# phys_params: numpy array
		trans_param = scale_phys_params(phys_params)
		with torch.no_grad():
			t_param = torch.tensor(trans_param, dtype=torch.float32).to(device)
			preds = model(t_param).cpu().numpy()
		return return_to_spectra(preds)

	def neural_network(trans_param):
		with torch.no_grad():
			t_param = torch.tensor(trans_param, dtype=torch.float32).to(device)
			preds = model(t_param).cpu().numpy()
		return preds

	return neural_network,neural_network_from_phys,params_cnst,spec_cnst


def load_model_for_sushi(model_path, device=None):
	"""
	Load a trained PyTorch surrogate model for use inside the SUSHI optimization loop.

	Unlike `load_model()`, this returns a **differentiable** callable:
	  model_callable(Theta: torch.Tensor) -> spectra: torch.Tensor
	where `Theta` is the (batch, N_params) normalized parameter tensor and the output
	is the (batch, n_channels) physical-space spectra.

	Gradients flow through the NN and the inverse PCA transform, allowing PyTorch 
	autograd to compute ∂cost/∂Theta via a single loss.backward() call.

	RETURNS
	-------
	model_callable  : differentiable callable (torch.Tensor -> torch.Tensor)
	params_cnst     : normalization constants for physical parameters
	spec_cnst       : normalization constants for spectra
	"""
	if device is None:
		if torch.backends.mps.is_available():
			device = torch.device("mps")
		elif torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

	with open(model_path, "rb") as f:
		model_dic = pickle.load(f)
		train_direc = model_dic["train_direc"]

	n_features = train_direc["X_train"].shape[1]
	n_targets  = train_direc["Y_train"].shape[1]

	nn_model = build_net(n_features, n_targets)
	if "model_state_dict" in model_dic:
		nn_model.load_state_dict(model_dic["model_state_dict"])
	elif "trained_params" in model_dic:
		raise ValueError("This model was trained with JAX. Load with load_model() instead.")
	else:
		raise ValueError("Invalid PyTorch model file.")

	nn_model.eval()
	nn_model.to(device)

	params_cnst = train_direc["params_cnst"]
	spec_cnst   = train_direc["spec_cnst"]

	# Cast PCA normalization constants to real and to torch tensors (on device)
	spec_log_mean_t  = torch.tensor(np.real(spec_cnst[0]), dtype=torch.float32, device=device)  # (n_channels,)
	spec_log_std_t   = torch.tensor(np.real(spec_cnst[1]), dtype=torch.float32, device=device)  # (n_channels,)
	U_spec_t         = torch.tensor(np.real(spec_cnst[2]), dtype=torch.float32, device=device)  # (n_channels, rank)
	spec_PCA_mean    = float(np.real(spec_cnst[3]))
	spec_PCA_std     = float(np.real(spec_cnst[4]))

	# Also cast params_cnst elements to real
	params_cnst = [np.real(c) for c in params_cnst]

	def model_callable(Theta: torch.Tensor) -> torch.Tensor:
		"""
		Differentiable forward pass: normalized params -> physical spectra.
		Theta : (batch, N_params) torch.Tensor  —  may have requires_grad=True
		Returns: (batch, n_channels) torch.Tensor
		"""
		# NN forward: normalized params -> PCA coefficients
		pca_out = nn_model(Theta)                                  # (batch, rank)
		# Inverse PCA normalization
		pca_phys = pca_out * spec_PCA_std + spec_PCA_mean          # (batch, rank)
		# Inverse PCA rotation: PCA -> log-normalized spectral space
		log_spec = pca_phys @ U_spec_t.T                           # (batch, n_channels)
		# Inverse log normalization
		log_spec = log_spec * spec_log_std_t + spec_log_mean_t     # broadcast over batch
		# Inverse log
		spec = torch.exp(log_spec)                                  # (batch, n_channels)
		return spec

	return model_callable, params_cnst, spec_cnst
