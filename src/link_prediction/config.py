import wandb
from torch_geometric.datasets import WebKB, Planetoid, WikipediaNetwork
# WebKB: (Texas, Wisconsin, Cornell); Planetoid: (Citeseer, Pubmed, Cora); WikipediaNetwork: (Squirrel, Chameleon)
print("Link prediction features initialized.....")


wb = True
project_name = 'Link Prediction with PBGNN'


dataset_diz = {
    'Texas': WebKB(root='/tmp/Texas', name='Texas'), # Texas is solved in Link prediction.
    'Cora': Planetoid(root='/tmp/Cora', name='Cora', split='Geom-GCN')
}


epochs = 5000
dataset_name = 'Texas'
dataset = dataset_diz[dataset_name]


batch_size = dataset.x.shape[0]
lr = 3e-3
wd = 0
num_layers = 2
hidden_dim = 256
step = 0.1
output_dim = 32
mlp_layer = 0
link_bias = False
dropout = 0.3

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


