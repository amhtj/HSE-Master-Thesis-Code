import os
import numpy as np
import torch
from torch import nn
from scipy import linalg
from tqdm import tqdm
import soundfile as sf
import resampy
from multiprocessing.dummy import Pool as ThreadPool
from .models.pann import Cnn14_16k


sample_rate = 16000


def load_signal_task(fname):
    wav_data, sr = sf.read(fname, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    if sr != sample_rate:
        wav_data = resampy.resample(wav_data, sr, sample_rate)

    return wav_data


class FrechetAudioDistance:
    def __init__(self, model_name="vggish", use_pca=False, use_activation=False, verbose=False, audio_load_worker=8):
        self.model_name = model_name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__get_model(model_name=model_name, use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
    
    def __get_model(self, model_name="vggish", use_pca=False, use_activation=False):
        
    
        if model_name == "vggish":
            
            self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            if not use_pca:
                self.model.postprocess = False
            if not use_activation:
                self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        
        elif model_name == "pann":
             
# Following the example from the paper cited [32] in our work, we take convolutional features and embeddings from :
            
            model_path = os.path.join(torch.hub.get_dir(), "Cnn14_16k_mAP%3D0.438.pth")
            if not(os.path.exists(model_path)):
                torch.hub.download_url_to_file('https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth', torch.hub.get_dir())
            self.model = Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])

        self.model.eval()
    
    def get_embeddings(self, x, sr=sample_rate):
        
        embd_lst = []
        try:
            for audio in tqdm(x, disable=(not self.verbose)):
                if self.model_name == "vggish":
                    embd = self.model.forward(audio, sr)
                elif self.model_name == "pann":
                    with torch.no_grad():
                        out = self.model(torch.tensor(audio).float().unsqueeze(0), None)
                        embd = out['embedding'].data[0]
                if self.device == torch.device('cuda'):
                    embd = embd.cpu()
                embd = embd.detach().numpy()
                embd_lst.append(embd)
        
        return np.concatenate(embd_lst, axis=0)
    
    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma
    
# There is nothing new in this block compared to the classical FID (used for GANs, VAEs, models like StyleGAN etc.), 
# so we take it from the official FID GitHub and embed it for our task :
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
     
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
    
    def __load_spectrum_files(self, dir):
        task_results = []

        pool = ThreadPool(self.audio_load_worker)
        pbar = tqdm(total=len(os.listdir(dir)), disable=(not self.verbose))

        def update(*a):
            pbar.update()

        if self.verbose:
            print("[Frechet Audio Distance] Loading audio from {}...".format(dir))
        for fname in os.listdir(dir):
            res = pool.apply_async(load_spectrum_task, args=(os.path.join(dir, fname),), callback=update)
            task_results.append(res)
        pool.close()
        pool.join()     

        return [k.get() for k in task_results] 

    def score(self, background_dir, eval_dir, store_embds=False):
        try:
            audio_background = self.__load_spectrum_files()_files(background_dir)
            audio_eval = self.__load_spectrum_files(eval_dir)

            embds_background = self.get_embeddings(signal_background)
            embds_eval = self.get_embeddings(signal_eval)

            if store_embds:
                np.save("embds_background.npy", embds_background)
                np.save("embds_eval.npy", embds_eval)

            if len(embds_background) == 0:
                print("[Frechet Audio Distance] background set dir is empty, exitting...")
                return -1
            if len(embds_eval) == 0:
                print("[Frechet Audio Distance] eval set dir is empty, exitting...")
                return -1
            
            mu_background, sigma_background = self.calculate_embd_statistics(embds_background)
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background, 
                sigma_background, 
                mu_eval, 
                sigma_eval
            )

            return fad_score
            
        except Exception as e:
            print("[Frechet Audio Distance] exception thrown, {}".format(str(e)))
            return -1