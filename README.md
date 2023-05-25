# HSE-Master-Thesis-Code
This repo dedicated to the master's thesis "Reconstruction of a low-amplitude signal with generative deep neural networks". As a developed model there are an alignment variational autoencoders.

How to run :
```
Download prepared dataset from Google Drive (https://drive.google.com/open?id=1ESXEmZLb20R-d8ok8n8wczhGpWduhaFx) and from CoREAS website
(https://www.huege.org/coreas/) ;
Run train.py python --signal cuted_signal.npy --noise cuted_noise.npy --min 100 --max 200 --epochs 100 --arch vae.py 
Get the model in .h5 format
```

The launch is possible only on a supercomputer due to the large volume of the data set and the complexity of the optimization process. 
