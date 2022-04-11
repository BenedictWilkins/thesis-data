# MNIST
python -m thesisdata -e thesis/MNIST-v0 -n 2 -a --policy 'gymu.policy.Uniform' --env_kwargs "{\"num_actions\":4,\"train\":false}" --test
python -m thesisdata -e thesis/MNIST-v0 -n 2 -a --policy 'gymu.policy.Uniform' --env_kwargs "{\"num_actions\":4}" --validate
python -m thesisdata -e thesis/MNIST-v0 -n 12 -a --policy 'gymu.policy.Uniform' --env_kwargs "{\"num_actions\":4}" --train
