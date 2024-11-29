python main.py --runner VAERunner --config vae/mnist_mcsm_forward_8.yml --doc mnist_mcsm_forward_vae_eps_1e-4_dim8_rademacher --device 6
python main.py --runner VAERunner --config vae/mnist_mcsm_forward.yml --doc mnist_mcsm_forward_vae_eps_1e-4_dim32_rademacher --device 6
# python main.py --runner WAERunner --config wae/celeba_mcsm_backward.yml --doc celeba_mcsm_backward_new_wae_eps_1e-3 --device 6
# python main.py --runner WAERunner --config wae/celeba_mcsm_backward.yml --doc celeba_mcsm_backward_new_wae_eps_1e-3 --device 6 --test