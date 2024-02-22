<img width="1286" alt="SUSHI_banner" src="https://github.com/JMLascar/SUSHI/assets/54967118/ef3bff3a-8656-4c34-b4cc-bd4fe56d931c">
This is the repository for the SUSHI (Semi-blind Unmixing with Sparsity for hyperspectral images) algorithm.

The purpose of SUSHI is to perform non-stationary unmixing of hyperspectral images.  

The typical use case explored in the companion paper is to map the physical parameters (e.g. temperature, redshift, etc) from a model with multiple components using data from hyperspectral images (aka integral field unit; IFUs).  
In order to obtain more robust results on voxels with low signal to noise ratio, a spatial regularization is applied. This enables to map the physical parameters at small scales without the need of a spatial rebinning. While the use cases explored in the paper is focused on X-ray astronomy, the method can be applied to any IFU data cubes. 

In this repositery, you will find:
* *SUSHI Test Notebook.ipynb*: A jupiter tutorial notebook to try out SUSHI on an example similar to that in [Lascar, Bobin, Acero, 2023].
* *SUSHI.py:* The SUSHI code itself, now able to take in a variable amount of components.
* *IAE_JAX_v2_devl_2023.py*: Code for the Interpolary Auto-Encoder (IAE), taken from: https://github.com/jbobin/IAE. 
* *Training an IAE.ipynb:* A tutorial notebook to train an IAE spectral model.
* *data:* repositery with the data sets needed for testing SUSHI.
* *IAE models:* readily trained IAE spectral models for testing SUSHI.
* *older_version:* Contains the Sushi algorithm in its previous architecture (where the number of components was hard-coded to 2; the current version takes any number of components) and the IAE code for older versions of JAX. 


# Tutorial: how to test SUSHI.
## Package requirements
SUSHI was coded in python 3.10. To test SUSHI, you will need the packages listed in SUSHI_env.yml. 
To create and activate a conda environment with all the imports needed, do:
- conda env create -f SUSHI_env.yml
- conda activate SUSHI_env

## Testing the sample example
Go to *SUSHI Test Notebook.ipynb* in the main directory. Normally, running the cells without further change should not present any issue.

## Testing on one's own data
### Training an IAE
To test SUSHI on one's own data, the first step will be to train IAE model(s) — one per physical component. The tutorial notebook will be in *Training an IAE.ipynb*. 
In that notebook, the training set is stored in the dictionary "Output". It should be constructed as such: 
* Provide a large number of spectra (1e3 at least) from your physical model. 
* Select the spectra that are at boundary parameters (e.g. highest and lowest temperature). These will be your anchor points: Output["Psi"].
* Shuffle the rest of the training set. Separate it into a training (70%) and validation (30%) set. Output["X_train"] and Output["X_valid"].
* These three sets (anchor points, training, and validation) should be numpy arrays of shape (N, $n_E$), where N is the number of spectra, and $n_E$ is the number of spectral channels.
* Now, you're ready for training. Parameters regarding the architecture can be customized in the last cell:
  - cost_weight: whether the training set should be weighted. Pondering by the mean was shown to be useful in cases of very dynamic ranges. Put "None" to avoid using.
  - niters: The training is done in three epochs (improving the model each time by taking the previous training as first guess). niters is a list containing the number of iterations per epoch.
  - opts: List containing the optimizer index from the list Optims = ['adam', 'momentum', 'rmsprop', 'adagrad', 'Nesterov', 'SGD'], for each epoch.
  - steps: List containing the step size for each epoch.
  - fname: the name of the model.
 
  Once the training is complete (this may take a while), check that the cost successfully converged.
[To be added: Testing of the IAE model]
Repeat this process for each physical component present in your model.

### Using SUSHI
From there, using the SUSHI testing notebook (*SUSHI Test Notebook (with variable number of components).ipynb*) should be fairly straightforward. In the cell that imports the models, replace the model_name by those of your trained IAE.
  
