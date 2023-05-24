The code for preprocessing our signals. More detailed :

__converter.py__

Used for convert from ADST to numpy binary format.

Arguments: 

* signal: signal file

* noise: noised signal file

Output: 2 numpy files


__creator.py__

Used for creation of training or test sets or change vectors dimension for neural network. Use "upsampling" for create dataset with augmentation.

Arguments:
* signal: signals
* noise: noised signals
* center: approximate peak position
* window: size of output numpy array
 
Output: 2 numpy files

__denoiser.py__

Used for denoising input file.

Arguments:
* noise: noised signals
* result: output file
* model: model for denoising

Output: 1 numpy file

__estimator.py__

Creates .csv table which summarizing result.

Arguments:
* true: true signals
* reco: reco signals
* upsampling: upsampling for transfer count to ns
* mode: snr or simple
