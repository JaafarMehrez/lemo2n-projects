import yaml

model_dict = {'precision': 32, 'num_chem_elements': 3, 'edge_cutoff': 4.5, 'num_edge_rbf': 8, 'num_edge_poly_cutoff': 6, 'vel_max': 0.11, 'num_vel_rbf': 8, 'max_rotation_order': 1, 'num_hidden_channels': 16, 'num_mp_layers': 3, 'edge_mlp_kwargs': {'n_neurons': [16, 16, 16], 'activation': 'silu'}, 'vel_mlp_kwargs': {'n_neurons': [16, 16, 16], 'activation': 'silu'}, 'nl_gate_kwargs': {'activation_scalars': {'o': 'tanh', 'e': 'silu'}, 'activation_gates': {'e': 'silu'}}, 'conserve_ang_mom': True, 'o3_backend': 'e3nn', 'net_lin_mom': [0.0, 0.0, 0.0], 'net_ang_mom': [0.0, 0.0, 0.0], 'avg_num_neighbors': 23.26775360107422}
    
data_dict = {"root": ".","name": "SiSiO2_train","cutoff_radius": 4.5,"files": ["./data/train.extxyz"],"rename": True,"atom_type_mapper":{14: 0,8: 1,1: 2,},}
 
train_dict = {'seed': 1705, 'model_type': 'EfficientTrajCastModel', 'device': 'cuda', 'restart_latest': True, 'target_field': 'target', 'reference_fields': ['displacements', 'update_velocities'], 'batch_size': 10, 'max_grad_norm': 0.5, 'num_epochs': 1500, 'criterion': {'loss_type': {'main_loss': 'mse'}, 'learnable_weights': False, 'dimensions': [3, 3]}, 'optimizer': 'adam', 'optimizer_settings': {'lr': 0.01, 'amsgrad': True}, 'scheduler': ['ReduceLROnPlateau'], 'scheduler_settings': {'ReduceLROnPlateau': {'factor': 0.8, 'patience': 25, 'min_lr': 0.0001}}, 'chained_scheduler_hp': {'milestones': [], 'per_epoch': True, 'monitor_lr_scheduler': False}, 'tensorboard_settings': {'loss': True, 'lr': True, 'loss_validation': {'data': {'root': '.', 'name': 'SiSiO2_val', 'cutoff_radius': 4.5, 'files': ['./data/val.extxyz'], 'rename': True, 'atom_type_mapper': {14: 0, 8: 1, 1: 2}}}}}
    

config = {"model": model_dict, "data": data_dict, "training": train_dict}

with open('manually_recreated_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Config recreated as 'manually_recreated_config.yaml'.")
