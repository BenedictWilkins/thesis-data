# Atari
python -m thesisdata -e SpaceInvadersNoFrameskip-v4 -n 50 -a --policy 'stable_baselines3.A2C' --train
python -m thesisdata -e SpaceInvadersNoFrameskip-v4 -n 20 -a --policy 'stable_baselines3.A2C' --validate
python -m thesisdata -e SpaceInvadersNoFrameskip-v4 -n 20 -a --policy 'stable_baselines3.A2C' --test


#python -m thesisdata -e thesis/MNIST-v0 -n 2 -a --policy 'stablebaselines3.A2C' --env_kwargs --test
#python -m thesisdata -e thesis/MNIST-v0 -n 2 -a --policy 'gymu.policy.Uniform' --env_kwargs "{\"num_actions\":4}" --validate
