import wandb
from torch_geometric.datasets import WebKB, Planetoid, WikipediaNetwork
# WebKB: (Texas, Wisconsin, Cornell); Planetoid: (Citeseer, Pubmed, Cora); WikipediaNetwork: (Squirrel, Chameleon)
print("Node classification features initialized.....")

wb = False
project_name = 'Link Prediction with PBGNN'


dataset_diz = {
    'Texas': WebKB(root='/tmp/Texas', name='Texas'), # Texas is solved in Link prediction.
    'Cora': Planetoid(root='/tmp/Cora', name='Cora', split='Geom-GCN'),
    'PubMed': Planetoid(root='/tmp/PubMed', name='PubMed', split='Geom-GCN'),
    'Wisconsin': WebKB(root='/tmp/Wisconsin', name = 'Wisconsin')
}

epochs = 15000
dataset_name = 'Cora'
dataset = dataset_diz[dataset_name]

batch_size = dataset.x.shape[0]
lr = 0.0026
wd = 0.0413
num_layers = 12
hidden_dim = 64
step = 0.25
loss_type = 'nll'


hyperparameters = {'batch_size': batch_size,
                   'learning rate': lr,
                   'weight decay': wd,
                   'Loss type': loss_type,
                   'N° Hidden layer': num_layers,
                   'Hidden dimension in GRAFF': hidden_dim,
                   'ODE step': step,
                   'Dataset': dataset_name}

sweep_config = {
    'method': 'random'
}
sweep_config['metric'] = {'name': 'Accuracy on test',
                          'goal': 'maximize'}

parameters_dict = {
    'lr': {
        'values': [1e-3, 1e-4, 1e-5]
    },
    'hidden_dim': {
        'values': [32, 64, 128, 256, 512, 1024]
    },
    'wd': {
        'values': [0, 1e-2, 1e-4, 1e-6]
    },
    'step': {
        'values': [0.1, 0.2, 0.3, 0.4, 0.5]
    },
    'num_layers': {
        'values': [1, 2, 3, 4]
    },
    'loss_type': {
        'values': ['ce', 'nll']
    }
}




sweep_config['parameters'] = parameters_dict


