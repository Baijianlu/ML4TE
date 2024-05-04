from __future__ import annotations

import os
import shutil
import warnings
import zipfile
from functools import partial
import re

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
from matgl.layers import BondExpansion, MLP, ActivationFunction
from matgl.models import MEGNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule, xavier_init
from matgl.config import DEFAULT_ELEMENTS

from d2l import torch as d2l
from torchinfo import summary
# To suppress warnings for clearer output
warnings.simplefilter("ignore")

class MP_dataset(d2l.DataModule):
    def __init__(self, dataset, batch_size=64, num_workers=1):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val, self.test = dataset[0], dataset[1], dataset[2]

        self.train_loader, self.val_loader = MGLDataLoader(
            train_data=self.train,
            val_data=self.val,
            test_data=self.test,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    def get_dataloader(self, train):
        if train:
            return self.train_loader
        else:
            return self.val_loader

    def visualize(self, batch):
        pass

class MyMEGNetClassifier(d2l.Module):
    def __init__(self, lr=0.1, elem_list=DEFAULT_ELEMENTS, dropout=0.0):
        super().__init__()
        self.save_hyperparameters()
        bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=6.0, num_centers=100, width=0.5)
        self.net = MEGNet(
            dim_node_embedding=16,
            dim_edge_embedding=100,
            dim_state_embedding=5,
            nblocks=3,
            hidden_layer_sizes_input=(8, 4),
            hidden_layer_sizes_conv=(8, 8, 4),
            nlayers_set2set=1,
            niters_set2set=2,
            hidden_layer_sizes_output=(8, 8),
            is_classification=True,
            activation_type="softplus2",
            bond_expansion=bond_expansion,
            cutoff=5.0,
            gauss_width=0.5,
            dropout=dropout,
            include_state=True
        )
        xavier_init(self.net)
    def forward(self, g, state_attr=None):
        node_feat = g.ndata["node_type"]
        edge_feat = g.edata["edge_attr"]
        return self.net(g, edge_feat, node_feat, state_attr)
        
    def loss(self, y_hat, y, averaged=True):
        y_hat = d2l.reshape(y_hat, (-1,))
        return F.binary_cross_entropy(y_hat, y, reduction='mean' if averaged else 'none')

    def accuracy(self, y_hat, y):
        y_hat = d2l.reshape(y_hat, (-1,))
        preds = d2l.astype(torch.tensor([1 if i >= 0.5 else 0 for i in y_hat]), y.dtype)
        compare = preds == d2l.reshape(y, -1)
        return float(d2l.reduce_sum(d2l.astype(compare, y.dtype)))
        
    def configure_optimizers(self):
#        return torch.optim.SGD(self.parameters(), self.lr, momentum=0.9)
        params_wonode = [param for name, param in self.net.named_parameters() if re.match('embedding.layer_node_embedding', name) is None]
        return torch.optim.Adam(params_wonode, lr=self.lr, eps=1e-8, weight_decay=0.0001)

    def configure_optimizers_finetuning(self):
        params_1x = [param for name, param in self.net.named_parameters() if re.match('output_proj', name) is None]
        return torch.optim.Adam([{'params': params_1x}, 
                                 {'params': self.net.output_proj.parameters(), 'lr': self.lr*10}], 
                                lr=self.lr, eps=1e-8)
        
    def configure_scheduler(self, optimizer, start_factor=1.0, end_factor=0.1, total_iters=499):
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
        
    def get_w_b(self):
        pass

    def training_step(self, batch):
        g, lat, state_attr, labels = batch
        l = self.loss(self(g=g, state_attr=state_attr), labels)
        return l, labels.shape[0]

    def validation_step(self, batch):
        g, lat, state_attr, labels = batch
        y_hat = self(g=g, state_attr=state_attr)
        l = self.loss(y_hat, labels)
        acc = self.accuracy(y_hat, labels)
        return l, acc, labels.shape[0]
    
    def predict_step(self, structures, state_attrs=None, converter=None):
        preds = list()
        state_attrs = torch.tensor(state_attrs).float()
        for struc,state_attr in zip(structures,state_attrs):
            state_attr = d2l.reshape(state_attr, (-1,5))
            prob = self.net.predict_structure(struc, state_feats=state_attr, graph_converter=converter).numpy()
            preds.append(prob)
        return preds

class MyTrainer(d2l.Trainer):
    def fit(self, model, data, dims, filepath, dropout=0.0):
        self.prepare_data(data)
        self.prepare_model_finetuning(model, dims, dropout=dropout)
        self.optim = model.configure_optimizers()
        self.scheduler = model.configure_scheduler(self.optim)
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        animator = d2l.Animator(xlabel='epoch', xlim=[0, self.max_epochs],
                            legend=['train_loss', 'val_loss', 'val_acc'], filepath=filepath)
        timer = d2l.Timer()
        for self.epoch in range(self.max_epochs):
            metric = d2l.Accumulator(5)
            self.model.train()
            for i, batch in enumerate(self.train_dataloader):
                timer.start()
                loss, batch_size = self.model.training_step(self.prepare_batch(batch))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                with torch.no_grad():
                    metric.add(loss * batch_size, batch_size, 0,0,0)
                train_loss = metric[0] / metric[1]
                timer.stop()
                if (i + 1) % (self.num_train_batches//2) == 0:
                    animator.add(self.epoch + (i+1) / self.num_train_batches,(train_loss, None, None))
            if self.val_dataloader is None:
                return
            self.model.eval()
            for batch in self.val_dataloader:
                with torch.no_grad():
                    loss_val, acc_val, batch_size = self.model.validation_step(self.prepare_batch(batch))
                    metric.add(0,0, loss_val * batch_size, acc_val, batch_size)
            val_loss = metric[2]/metric[4]
            val_acc = metric[3]/metric[4]
            animator.add(self.epoch+1, (None, val_loss, val_acc)) 
        
            self.scheduler.step()
        
        training_log = {'train_loss': (animator.X[0], animator.Y[0]), 
                       'val_loss': (animator.X[1], animator.Y[1]),
                       'val_acc': (animator.X[2], animator.Y[2])}
        print(training_log)
        print(f'Total training time: {timer.sum():.1f}')

    def fit_fine_tuning(self, orig_model, data, dims, filepath, dropout=0.0):
        self.prepare_data(data)
        self.prepare_model_finetuning(orig_model, dims, dropout=dropout)
        self.optim = orig_model.configure_optimizers_finetuning()
        self.scheduler = orig_model.configure_scheduler(self.optim, decay_alpha=0.1)
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        animator = d2l.Animator(xlabel='epoch', xlim=[0, self.max_epochs],
                            legend=['train_loss', 'val_loss', 'val_acc'], filepath=filepath)
        timer = d2l.Timer()
        for self.epoch in range(self.max_epochs):
            metric = d2l.Accumulator(5)
            self.model.train()
            for i, batch in enumerate(self.train_dataloader):
                timer.start()
                loss, batch_size = self.model.training_step(self.prepare_batch(batch))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                with torch.no_grad():
                    metric.add(loss * batch_size, batch_size, 0,0,0)
                train_loss = metric[0] / metric[1]
                timer.stop()
                if (i + 1) % (self.num_train_batches//2) == 0:
                    animator.add(self.epoch + (i+1) / self.num_train_batches,(train_loss, None, None))
            if self.val_dataloader is None:
                return
            self.model.eval()
            for batch in self.val_dataloader:
                with torch.no_grad():
                    loss_val, acc_val, batch_size = self.model.validation_step(self.prepare_batch(batch))
                    metric.add(0,0, loss_val * batch_size, acc_val, batch_size)
            val_loss = metric[2]/metric[4]
            val_acc = metric[3]/metric[4]
            animator.add(self.epoch+1, (None, val_loss, val_acc)) 
        
            self.scheduler.step()
        
        training_log = {'train_loss': (animator.X[0], animator.Y[0]), 
                       'val_loss': (animator.X[1], animator.Y[1]),
                       'val_acc': (animator.X[2], animator.Y[2])}
        print(training_log)
        print(f'Total training time: {timer.sum():.1f}')
        
    def prepare_model_finetuning(self, model, dims, dropout=0.0, activation_type="softplus2"):
        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None
        model.net.output_proj = MLP(
                                    dims=dims,
                                    activation=activation,
                                    activate_last=False,
                                    dropout=dropout,
                                    )
        xavier_init(model.net.output_proj)
        model.net.is_classification = True
        model.net.dropout = nn.Dropout(dropout) if dropout else None 
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

def ClearFiles():
    for fn in ("dgl_graph.bin", "dgl_line_graph.bin", "state_attr.pt", "labels.json", "lattice.pt"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    '''Comments '''
    df1 = pd.read_excel("X_train_n.xlsx", sheet_name='Sheet1',index_col=False)
    df2 = pd.read_excel("X_test_n.xlsx", sheet_name='Sheet1',index_col=False)
    train_structures = [Structure.from_str(struct, 'json') for struct in df1['structure_opt']]
    val_structures = [Structure.from_str(struct, 'json') for struct in df2['structure_opt']]
    train_labels = df1['class'].tolist()
    val_labels = df2['class'].tolist()
    train_state = df1.loc[:, ['N_c','m_cond_c','m_dos_c','bandgap','defconst_c']].to_numpy()
    val_state = df2.loc[:, ['N_c','m_cond_c','m_dos_c','bandgap','defconst_c']].to_numpy()

    converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=5.0)
  
    map_location = torch.device("cpu") if not torch.cuda.is_available() else None 
    model = MyMEGNetClassifier(lr=0.001, dropout=0.0)
    dims = [21,8,8,1]
    trainer = MyTrainer(max_epochs=10, num_gpus=0)
    trainer.prepare_model_finetuning(model, dims, dropout=0.0)
    model.net.load_state_dict(torch.load('state.pt', map_location=map_location))
    model.net.bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=6.0, num_centers=100, width=0.5)

    model.eval()
    preds = model.predict_step(val_structures, state_attrs=val_state, converter=converter)
    Y_hat = [1 if i >= 0.5 else 0 for i in preds]
    df2['prob'] = pd.Series(preds, dtype=float)
    df2['pred'] = pd.Series(Y_hat, dtype=int)
    df2.to_excel('X_test_n_withpreds.xlsx', index_label='index', merge_cells=False)
