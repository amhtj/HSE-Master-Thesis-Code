Signal preprocessing. A brief description of the work sequence :

**converter.py**
Used for convert from ADST to numpy binary format.
Arguments:
signal: signal file
noise: noised signal file
Output: 2 numpy files

**creator.py**
Used for creation of training or test sets or change vectors dimension for neural network. Use "upsampling" for create dataset with augmentation.
Arguments:
signal: signals
noise: noised signals
center: approximate peak position
window: size of output numpy array
Output: 2 numpy files

**denoiser.py**
Used for denoising input file.
Arguments:
noise: noised signals
result: output file
model: model for denoising
Output: 1 numpy file

**estimator.py**
Creates .csv table which summarizing result.
Arguments:
true: true signals
reco: reco signals
upsampling: upsampling for transfer count to ns
mode: snr or simple
Output: 1 csv file
