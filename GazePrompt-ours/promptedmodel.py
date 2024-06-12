import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from features import resnet18
from functools import reduce
import pdb
from functools import reduce
from operator import mul
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, embed_dim, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.embed_dim = embed_dim

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):

        q = k = src
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7*7
        dim_feedforward=512
        dropout = 0.1
        num_layers=6

        self.base_model = resnet18(pretrained=False, maps=maps)

        encoder_layer = TransformerEncoderLayer(
                  maps, 
                  nhead, 
                  dim_feedforward, 
                  dropout)

        encoder_norm = nn.LayerNorm(maps) 

        self.encoder = TransformerEncoder(encoder_layer, num_layers, dim_feedforward, encoder_norm)
    
        # ADDITION OF PROMPT
        self.prompt_tokens = 5  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(0.0)
        self.prompt_dim = maps
        self.prompt_type = "shallow" # "shallow" or "deep"
        assert self.prompt_type in ['shallow','deep'], "prompt type should be 'shallow' or 'deep'."

        val = math.sqrt(6. / (self.prompt_tokens+self.prompt_dim))  

        self.prompt_embeddings = nn.Parameter(torch.zeros(self.prompt_tokens, 1,self.prompt_dim))

        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if self.prompt_type == 'deep':  
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(self.prompt_tokens,num_layers-1, self.prompt_dim)
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        self.feed = nn.Linear(maps, 2)
            
        self.loss_op = nn.L1Loss()
        #pdb.set_trace()

    def incorporate_prompt(self, x, prompt_embeddings, n_prompt: int = 0):
        B = x.shape[0]
        #pdb.set_trace()
        x = torch.cat((
            x[:1, :, :],
            self.prompt_dropout(prompt_embeddings),
            x[(1+n_prompt):,:,:]
        ), dim=0)
        
        return x

    def forward_deep_prompt(self, embedding_output):
        B = embedding_output.shape[0]

    def forward(self, x_in):
        #pdb.set_trace()
        feature = self.base_model(x_in["face"])
        batch_size = feature.size(0)
        #pdb.set_trace()
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)

# ADDED PROMPT

        prompt_embeddings = self.prompt_embeddings.repeat((1, batch_size, 1))
        x = self.incorporate_prompt(feature, prompt_embeddings,5)
        
        if self.prompt_type == 'deep':
            # deep mode
            for i in range(1, 6):
                x = self.incorporate_prompt(x, self.deep_prompt_embeddings[i-1], self.prompt_tokens)
                x = self.encoder[i](x)
        else:
            # shallow mode
            x = self.encoder(x)

        feature=x
        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0]

        gaze = self.feed(feature)
        
        return gaze

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label) 
        return loss
    
    def train(self, mode=True):
        if mode:
            self.base_model.eval()
            self.encoder.eval()
            self.feed.eval()
            self.prompt_embeddings.requires_grad = False
            # Only train the prompt-related parts
            for module in [self.prompt_dropout]:
                module.train()
        else:
            super().train(mode)
    ### END

