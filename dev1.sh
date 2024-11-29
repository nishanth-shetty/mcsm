python main.py --runner VAERunner --config vae/mnist_mcsm_forward_8.yml --doc mnist_mcsm_forward_vae_eps_1e-7_dim8 --device 0
python main.py --runner VAERunner --config vae/mnist_mcsm_forward.yml --doc mnist_mcsm_forward_vae_eps_1e-7_dim32 --device 0
python main.py --runner VAERunner --config vae/mnist_mcsm_forward_8.yml --doc mnist_mcsm_forward_vae_eps_1e-7_dim8 --device 0 --test
python main.py --runner VAERunner --config vae/mnist_mcsm_forward.yml --doc mnist_mcsm_forward_vae_eps_1e-7_dim32 --device 0 --test