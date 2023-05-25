Implementation of the VAE model from Section 4.2 of our work. Let's take a look at our architecture :

<img src="/model/architecture.jpg" alt="Alt text" width=50% height=50%>

Variational inference aims at finding the true conditional
probability distribution $\mathbf{p_\theta(x \mid z)}$  over the latent variables. The parameters such as $p_\theta$ are presumed on a VAE's block called a decoder, parameterized by \mathbf{$\theta$}, and $\mathbf{z}$ is taken as input.

The training procedure of our VAE contains low-bound maximization, called an Evidence Lower BOund (ELBO), and consist the Kullback-Leibler (KL) divergence. Our model requires computing the KL divergence between two multivariate Gaussians. Our architecture contains two identical autoencoders intersecting with each other. The encoder predicts $\mu$ and $\Sigma$ such that $\mathbf{q_\phi(z|x) = \mathcal{N} (\mu, \Sigma)}$. 

Our $f_i$ marked the signal variation coefficients matching with the energy, and $\mathcal{D}i$ marked an artificially-generated dataset of signal vectors, synthesized by changing $f_i$.

This structure of model helps to divide the training on Tunka-Rex data and on CoREAS data, the distributions obtained from the two autoencoders are coordinated with each other to create a latent space, and then the latent spaces are aligned according to the ELBO. To train the model, the parameters of both the generative model and the inference model are jointly evaluated using ELBO maximization. 



