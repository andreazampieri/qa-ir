import torch
import torch.nn as nn
import torch.nn.functional as f

from qair.models import Model
from qair.models.layers import KimConv, activations, AttentionMatrix, SimpleConv

torch.backends.cudnn.deterministic = True

@Model.register('base-cnn')
class CNN(Model):

    def __init__(self, params, vocab, device='cpu'):
        super().__init__()
        self.vocab = vocab

        self.device = torch.device(device)
        params['emb_num'], params['emb_dim'] = vocab.shape

        self.embs = nn.Embedding(params['emb_num'],
                                 params['emb_dim'],
                                 vocab['PAD'])

        self.conv_q = KimConv(params['emb_dim'],
                              params['qcnn']['conv_size'],
                              windows=[params['qcnn']['window']],
                              activation=activations[params['qcnn']['activation']])

        self.conv_a = KimConv(params['emb_dim'],
                            params['acnn']['conv_size'],
                            windows=[params['acnn']['window']],
                            activation=activations[params['acnn']['activation']])

        self.embs.weight.data.copy_(torch.from_numpy(vocab.weights))
        if 'static_emb' in params and params['static_emb']:
            self.embs.weight.requires_grad = False

    def forward(self, inp):
        (q, a) = inp
        # Map ids to word embeddings
        q = self.embs(q)
        a = self.embs(a)

        # Convolutional Encoder

        qemb = self.conv_q(q)
        aemb = self.conv_a(a)

       	out = f.cosine_similarity(qemb,aemb)

        return out

@Model.register('att-cnn')
class AttCNN(Model):

    def __init__(self, params, vocab, device='cpu'):
        super().__init__()
        self.vocab = vocab

        self.device = torch.device(device)
        params['emb_num'], params['emb_dim'] = vocab.shape

        self.embs = nn.Embedding(params['emb_num'],
                                 params['emb_dim'],
                                 vocab['PAD'])

        self.conv_q = SimpleConv(params['emb_dim'],
                                 params['qcnn']['conv_size'],
                                 params['qcnn']['window'],
                                 activation=activations[params['qcnn']['activation']])

        self.conv_a = SimpleConv(params['emb_dim'],
                                 params['acnn']['conv_size'],
                                 params['acnn']['window'],
                                 activation=activations[params['acnn']['activation']])

        self.h_pool = lambda x: self.horizontal_pooling(x)
        self.v_pool = lambda x: self.vertical_pooling(x)

        self.att = AttentionMatrix(params['qcnn']['conv_size'])

        self.embs.weight.data.copy_(torch.from_numpy(vocab.weights))
        if 'static_emb' in params and params['static_emb']:
            self.embs.weight.requires_grad = False

    def horizontal_pooling(self,t):
        return f.max_pool1d(t,t.size(2)).view(t.size(0),-1)

    def vertical_pooling(self,x):
        return self.horizontal_pooling(x.transpose(1,2)) 

    def flatten(self,x):
        return x.view(x.size(0),-1)

    def forward(self, inp):
        (q, a) = inp
        # Map ids to word embeddings
        q = self.embs(q)
        a = self.embs(a)

        # Convolutional Encoder

        qemb = self.conv_q(q)
        aemb = self.conv_a(a)

        mat = self.att(qemb,aemb)
    

        q_att = f.softmax(self.h_pool(mat),dim=1)
        a_att = f.softmax(self.v_pool(mat),dim=1)
        q = self.flatten(torch.matmul(qemb,q_att))
        a = self.flatten(torch.matmul(aemb,a_att))

        return f.cosine_similarity(q,a)

# ---------------------------------------

import numpy as np

class AttentionMatrix(nn.Module):
  
    def __init__(self,emb_dim):
        super(AttentionMatrix,self).__init__()
        #self.u = nn.Parameter(torch.Tensor(emb_dim,emb_dim).type(torch.FloatTensor),requires_grad=True)
        self.u = nn.Parameter(torch.from_numpy(np.random.normal(size=(emb_dim,emb_dim))).type(torch.FloatTensor),requires_grad=True)
    
    def forward(self,q,a):
        qt = q.transpose(1,2)
        out = torch.matmul(torch.matmul(qt,self.u),a) # Qt*U*A
        return torch.tanh(out)

class ConvolutionModule(nn.Module):
  
    def __init__(self,emb_dim,hidden_dim,ctx_window):
        super(ConvolutionModule,self).__init__()
        self.conv = nn.Conv2d(1,hidden_dim,kernel_size=(ctx_window,emb_dim))
        odd_adjustment = 1 if ctx_window%2==0 else 0
        self.pad = nn.ZeroPad2d((0,0,ctx_window-1-odd_adjustment,ctx_window-1))
    
    def forward(self,x_emb):
        x_emb = x_emb.unsqueeze(1) #adding a dimension (the channel for the convolution)   
    #return f.relu(self.conv(self.pad(x_emb))).squeeze(dim=3) # remove the single channel extra dimension
        return torch.sigmoid(self.conv(self.pad(x_emb)).squeeze(dim=3))
    #return f.dropout(torch.sigmoid(self.conv(self.pad(x_emb)).squeeze(dim=3)),p=0.8)


@Model.register("test-att")
class AP_CNN(Model):
  
    def __init__(self, params, vocab, device='cpu'):
        super().__init__()
        self.vocab = vocab

        self.device = torch.device(device)
        params['emb_num'], params['emb_dim'] = vocab.shape

        self.embs = nn.Embedding(params['emb_num'],
                                 params['emb_dim'],
                                 vocab['PAD'])
        #self.convolution_q = ConvolutionModule(emb_dim,dict_size,hidden_dim,ctx_window)
        #self.convolution_a = ConvolutionModule(emb_dim,dict_size,hidden_dim,ctx_window)
        self.convolution = ConvolutionModule(params['emb_dim'],params['qcnn']['conv_size'],params['qcnn']['window'])
        self.attention_mat = AttentionMatrix(params['qcnn']['conv_size'])
        self.h_pool = lambda t : self.horizontal_pooling(t)
        self.v_pool = lambda t : self.vertical_pooling(t)
    
    def flatten(self,x):
      return x.view(x.size(0),-1)
    
    def forward(self,q,a):
        q_emb = self.emb(q)
        a_emb = self.emb(a)
        q_enc = self.convolution(q_emb)
        a_enc = self.convolution(a_emb)
        mat = self.attention_mat(q_enc,a_enc) # check dimensions
        q_att = f.softmax(self.h_pool(mat),dim=1)
        a_att = f.softmax(self.v_pool(mat),dim=1)
        q = self.flatten(torch.matmul(q_enc,q_att))
        a = self.flatten(torch.matmul(a_enc,a_att))
        return f.cosine_similarity(q,a)
      
    def horizontal_pooling(self,x):
        return f.max_pool1d(x,x.size(2))
  
    def vertical_pooling(self,x):
        return self.horizontal_pooling(x.transpose(1,2)) 
