python main.py --runner WAERunner --config wae/mnist_mcsm_8.yml --doc mnist_mcsm_central_wae_eps_1e-4_dim8_rademacher --device 7
python main.py --runner WAERunner --config wae/mnist_mcsm.yml --doc mnist_mcsm_central_wae_eps_1e-4_dim32_rademacher --device 7
# python main.py --runner WAERunner --config wae/mnist_mcsm_8.yml --doc mnist_mcsm_central_wae_eps_1e-7_dim8 --device 6 --test
# python main.py --runner WAERunner --config wae/mnist_mcsm.yml --doc mnist_mcsm_central_wae_eps_1e-7_dim32 --device 6 --test