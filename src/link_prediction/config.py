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


SEED = 21022


epochs = 5000
dataset_name = 'Texas'
dataset = dataset_diz[dataset_name]

batch_size = dataset.x.shape[0]

# lr = 0.0001
# wd = 0.001
# num_layers = 1
# hidden_dim = 128
# step = 0.1
# output_dim = 16
# mlp_layer = 1
# link_bias = False
# dropout = 0.2
# input_dropout = 0.4



# Wisconsin ROBUST AUROC
# lr = 0.0001
# wd = 0.001
# num_layers = 1
# hidden_dim = 128
# step = 0.1
# output_dim = 16
# mlp_layer = 1
# link_bias = False
# dropout = 0.2
# input_dropout = 0.4



# TEXAS ROBUST AUROC
# learning rate and number of layers are determining
# as in Wisconsin
# lunar_sweep
lr = 0.001
wd = 0.000001
num_layers = 1
hidden_dim = 256
step = 0.1
output_dim = 64
mlp_layer = 0
link_bias = False
dropout = 0.4
input_dropout = 0
runs = 10

# Homophilic datasets
# lr should be 0.01 num_layers small

# Heterophilic datasets
# lr and number of layers should be small. 
# Among the best in Wisconsin, output_dim and mlp_layer (!=0) become important.
# Among the best in Texas, input_dropout and mlp_layer (~0) become important.

# Wisconsin is bad


hyperparameters = {'batch_size': batch_size,
                   'learning rate': lr,
                   'weight decay': wd,
                   'output_dim': output_dim,
                   'mlp_layers': mlp_layer,
                   'link_bias': link_bias,
                   'decoder dropout': dropout,
                   'encoder dropout': input_dropout,
                   'N° Hidden layer': num_layers,
                   'Hidden dimension in GRAFF': hidden_dim,
                   'ODE step': step,
                   'runs per conf.': runs,
                   'Dataset': dataset_name}


# Set sweep = True in the notebook when is required to do the hyperparameter tuning with sweep #
sweep_config = {
    'method': 'random'
}
sweep_config['metric'] = {'name': 'AUROC on test (Mean)',
                          'goal': 'maximize'
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
        'values': [0.1, 0.2, 0.3, 0.5]
    },
    'num_layers': {
        'values': [1, 2, 3, 5, 7, 9, 12]
    },
    'output_dim': {
        'values': [16, 32, 64]
    },
    'mlp_layer': {
        'values': [0, 1, 2]
    },
    'dropout': {
        'values': [0, 0.2, 0.3, 0.4]
    },
    'input_dropout': {
        'values': [0, 0.2, 0.3, 0.4]
    }
}


# parameters_dict = {
#     'mlp_layer': {
#         'values': [0, 1, 2, 3, 4]
#     }
# }





sweep_config['parameters'] = parameters_dict


