# Monte Carlo Score Matching

This repo contains a PyTorch implementation for the ICASSP 2024 submissiion: Monte Carlo Score-Matching. 



## Dependencies

The following are packages needed for running this repo.

- PyTorch==1.0.1
- TensorFlow==1.12.0
- tqdm
- tensorboardX
- Scipy
- PyYAML



## Running the experiments
```bash
python main.py --runner [runner name] --config [config file]
```

Here `runner name` is one of the following:

- `DKEFRunner`. This corresponds to experiments on deep kernel exponential families.
- `NICERunner`. This corresponds to the sanity check experiment of training a NICE model.
- `VAERunner`. Experiments on VAEs.
- `WAERunner`. Experiments on Wasserstein Auto-Encoders (WAEs).

and `config file` is the directory of some YAML file in `configs/`.



For example, if you want to train an implicit VAE of latent size 8 on MNIST with Sliced Score Matching, just run

```bash
python main.py --runner VAERunner --config vae/mnist_ssm_8.yml
```
