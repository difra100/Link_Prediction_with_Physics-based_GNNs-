import wandb
from torch_geometric.datasets import WebKB, Planetoid, WikipediaNetwork
# WebKB: (Texas, Wisconsin, Cornell); Planetoid: (Citeseer, Pubmed, Cora); WikipediaNetwork: (Squirrel, Chameleon)
print("Link prediction features initialized.....")


wb = False
project_name = 'Link Prediction with PBGNN'


dataset_diz = {
    'Texas': WebKB(root='/tmp/Texas', name='Texas'), # Texas is solved in Link prediction.
    'Cora': Planetoid(root='/tmp/Cora', name='Cora', split='Geom-GCN'),
    'PubMed': Planetoid(root='/tmp/PubMed', name='PubMed', split='Geom-GCN'),
    'Wisconsin': WebKB(root='/tmp/Wisconsin', name = 'Wisconsin')
}


epochs = 5000
dataset_name = 'Wisconsin'
dataset = dataset_diz[dataset_name]


batch_size = dataset.x.shape[0]
lr = 0.0001
wd = 0.0001
num_layers = 2
hidden_dim = 64
step = 0.5
output_dim = 32
mlp_layer = 0
link_bias = True
dropout = 0.2
# GRAFF ~ 0.86

# Cora Dataset: GRAFF
# lr = 0.01           
# wd = 0.001
# num_layers = 2
# hidden_dim = 128
# step = 0.1
# output_dim = 32
# mlp_layer = 0
# link_bias = True
# dropout = 0.2
# GRAFF ~ 0.86

hyperparameters = {'batch_size': batch_size,
                   'learning rate': lr,
                   'weight decay': wd,
                   'output_dim': output_dim,
                   'mlp_layers': mlp_layer,
                   'link_bias': link_bias,
                   'dropout level': dropout,
                   'N° Hidden layer': num_layers,
                   'Hidden dimension in GRAFF': hidden_dim,
                   'ODE step': step,
                   'Dataset': dataset_name}


# Set sweep = True in the notebook when is required to do the hyperparameter tuning with sweep #
sweep_config = {
    'method': 'random'
}
sweep_config['metric'] = {'name': 'HR@100 on test',
                          'goal': 'maximize',
                         }

parameters_dict = {
    'lr': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'hidden_dim': {
        'values': [32, 64, 128, 256]
    },
    'wd': {
        'values': [0, 1e-2, 1e-3, 1e-6]
    },
    'step': {
        'values': [0.1, 0.2, 0.3]
    },
    'num_layers': {
        'values': [1, 2, 3]
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'mlp_layer': {
        'values': [0, 1, 2, 3]
    },
    'link_bias': {
       'values': [True, False]
    },
    'dropout': {
        'values': [0, 0.2, 0.3, 0.4]
    }
}


# parameters_dict = {
#     'lr': {
#         'values': [1e-2]
#     },
#     'hidden_dim': {
#         'values': [32, 64, 128, 256]
#     },
#     'wd': {
#         'values': [1e-3]
#     },
#     'step': {
#         'values': [0.1]
#     },
#     'num_layers': {
#         'values': [2]
#     },
#     'output_dim': {
#         'values': [32]
#     },
#     'mlp_layer': {
#         'values': [0, 1, 2]
#     },
#     'link_bias': {
#        'values': [True]
#     },
#     'dropout': {
#         'values': [0.2, 0.3]
#     }
# }







sweep_config['parameters'] = parameters_dict


