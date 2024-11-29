python main.py --runner WAERunner --config wae/mnist_mcsm_backward_8.yml --doc mnist_mcsm_backward_wae_eps_1e-3_dim8 --device 5
#python main.py --runner WAERunner --config wae/mnist_mcsm_backward.yml --doc mnist_mcsm_backward_wae_eps_1e-7_dim32 --device 5
python main.py --runner WAERunner --config wae/mnist_mcsm_backward_8.yml --doc mnist_mcsm_backward_wae_eps_1e-3_dim8 --device 5 --test
#python main.py --runner WAERunner --config wae/mnist_mcsm_backward.yml --doc mnist_mcsm_backward_wae_eps_1e-7_dim32 --device 5 --test