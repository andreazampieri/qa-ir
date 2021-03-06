import torch
from torch import nn
import numpy as np


activations = {
    'relu':torch.relu,
    'tanh':torch.tanh,
    'sigmoid':torch.sigmoid,
    None:None
}

def attention(q, a, mask_q=None, mask_a=None):
    dot_qa = torch.bmm(q, a.transpose(1, 2))

    if mask_q is not None:
        dot_qa.masked_fill_(mask_q.unsqueeze(2).expand_as(dot_qa),
                            float(-10**10))
    if mask_a is not None:
        dot_qa.masked_fill_(mask_a.unsqueeze(1).expand_as(dot_qa),
                            float(-10**10))
    att_qa = torch.softmax(dot_qa, dim=2)
    att_aq = torch.softmax(dot_qa, dim=1).transpose(1, 2)

    q_att = torch.bmm(att_qa, a)
    a_att = torch.bmm(att_aq, q)

    return q_att, a_att

class AttentionMatrix(nn.Module):
  
    def __init__(self,emb_dim):
        super(AttentionMatrix,self).__init__()
        self.u = nn.Parameter(torch.from_numpy(np.random.normal(size=(emb_dim,emb_dim))).type(torch.FloatTensor),requires_grad=True)
    
    def forward(self,q,a):
        qt = q.transpose(1,2)
        out = torch.matmul(torch.matmul(qt,self.u),a) # Qt*U*A
        return torch.tanh(out)

class SimpleConv(nn.Module):

    def __init__(self,input_dim,hidden_dim,ctx_window,activation=torch.relu):
        super(SimpleConv,self).__init__()
        self.conv = nn.Conv2d(1,hidden_dim,kernel_size=(ctx_window,input_dim))
        odd_adjustment = 1 if ctx_window%2==0 else 0
        self.pad = nn.ZeroPad2d((0,0,ctx_window-1-odd_adjustment,ctx_window-1))
        self.activation = activation
    
    def forward(self,x):
        x = x.unsqueeze(1) #adding a dimension (the channel for the convolution)   
        #return f.relu(self.conv(self.pad(x_emb))).squeeze(dim=3) # remove the single channel extra dimension
        return self.activation(self.conv(self.pad(x)).squeeze(dim=3))

class BiLSTM(nn.Module):
  
    def __init__(self,input_dim,single_hidden_dim,activation = None):
        super(BiLSTM,self).__init__()
        self.bilstm = nn.LSTM(input_dim,single_hidden_dim,bidirectional=True)
        self.activation = activation
    
    def forward(self,x):
        # LSTM returns a tuple, the tensor is the first element
        out = self.bilstm(x)[0]
        if self.activation  != None:
            out = self.activation(out)
        return out

def sigmoid_attention(q, a, mask_q=None, mask_a=None):
    dot_qa = torch.bmm(q, a.transpose(1, 2))

    if mask_q is not None:
        dot_qa.masked_fill_(mask_q.unsqueeze(2).expand_as(dot_qa),
                            float(-10**10))
    if mask_a is not None:
        dot_qa.masked_fill_(mask_a.unsqueeze(1).expand_as(dot_qa),
                            float(-10**10))
    att_qa = torch.sigmoid(dot_qa)
    att_aq = torch.sigmoid(dot_qa).transpose(1, 2)

    q_att = torch.bmm(att_qa, a)
    a_att = torch.bmm(att_aq, q)

    return q_att, a_att


def batched_cosine(q, a, mask_q=None, mask_a=None):
    dot = torch.bmm(q, a.transpose(1, 2))
    q_norms = torch.norm(q, 2, dim=2)
    a_norms = torch.norm(a, 2, dim=2)
    norms = torch.bmm(q_norms.unsqueeze(2), a_norms.unsqueeze(1))
    scores = torch.div(dot, norms.clamp(min=1e-8))
    if mask_q is not None:
        scores.masked_fill_(mask_q.unsqueeze(2).expand_as(scores),
                            float(0))
    if mask_a is not None:
        scores.masked_fill_(mask_a.unsqueeze(1).expand_as(scores),
                            float(0))
    q_o, _ = scores.max(dim=2)
    a_o, _ = scores.transpose(1, 2).max(dim=2)
    return q_o.unsqueeze(2), a_o.unsqueeze(2)

class Bottle(nn.Module):

    def forward(self, x):
        if len(x.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = x.size()[:2]
        out = super(Bottle, self).forward(x.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleLinear(Bottle, nn.Linear):
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    pass


class KimConv(nn.Module):

    def __init__(self, inp_dim, out_dim, windows=None, activation=torch.relu):
        super(KimConv, self).__init__()
        if windows is None:
            windows = [5]
        self.convs = nn.ModuleList([nn.Conv1d(inp_dim,
                                              out_dim,
                                              w,
                                              padding=w//2)
                                    for w in windows])
        self.activation = activation

    def forward(self, x):
        x = x.transpose(1, 2)
        mat = [conv(x).max(dim=2)[0] for conv in self.convs]
        xemb = torch.cat(mat, 1)
        return self.activation(xemb)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (self.gamma / (std + self.eps)) * (x - mean) + self.beta


class BottleLayerNorm(Bottle, LayerNorm):
    pass
