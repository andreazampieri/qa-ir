import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import Conv2d

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
                              windows=params['qcnn']['window'],
                              activation=activations[params['qcnn']['activation']])

        self.conv_a = KimConv(params['emb_dim'],
                            params['acnn']['conv_size'],
                            windows=params['acnn']['window'],
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
        self.convolution = SimpleConv(params['emb_dim'],params['qcnn']['conv_size'],params['qcnn']['window'])
        self.attention_mat = AttentionMatrix(params['qcnn']['conv_size'])
        self.h_pool = lambda t : self.horizontal_pooling(t)
        self.v_pool = lambda t : self.vertical_pooling(t)
        self.dense = nn.Sequential(
            nn.Linear(2*params['qcnn']['conv_size'], params['hidden_size']),
            nn.Tanh(),
            nn.Linear(params['hidden_size'], 1),
        )
    
    def flatten(self,x):
      return x.view(x.size(0),-1)
    
    def forward(self,inp):
        q,a = inp
        q_emb = self.embs(q)
        a_emb = self.embs(a)
        q_enc = self.convolution(q_emb)
        a_enc = self.convolution(a_emb)
        mat = self.attention_mat(q_enc,a_enc) # check dimensions
        q_att = f.softmax(self.h_pool(mat),dim=1)
        a_att = f.softmax(self.v_pool(mat),dim=1)
        q = self.flatten(torch.matmul(q_enc,q_att))
        a = self.flatten(torch.matmul(a_enc,a_att))
        #return f.cosine_similarity(q,a)

        out = self.dense(torch.cat([q,a],-1))
        return out
      
    def horizontal_pooling(self,x):
        return f.max_pool1d(x,x.size(2))
  
    def vertical_pooling(self,x):
        return self.horizontal_pooling(x.transpose(1,2)) 

@Model.register("deep-cnn")
class DeepCNN(Model):

    def __init__(self,params,vocab,device="cpu"):
        super().__init__()
        self.vocab = vocab

        self.device = torch.device(device)
        params['emb_num'], params['emb_dim'] = vocab.shape

        self.embs = nn.Embedding(params['emb_num'],
                                 params['emb_dim'],
                                 vocab['PAD'])

        # self.conv_1 = SimpleConv(params['emb_dim'],
        #                          params['cnn1']['conv_size'],
        #                          params['cnn1']['window'],
        #                          activation=activations[params['cnn1']['activation']])

        # self.conv_2 = SimpleConv(params['emb_dim'],
        #                          params['cnn2']['conv_size'],
        #                          params['cnn2']['window'],
        #                          channels=params['cnn1']['conv_size'],
        #                          activation=activations[params['cnn2']['activation']])

        emb_size = params['emb_dim']
        conv_1_filters = params['cnn1']['conv_size']
        conv_1_window = params['cnn1']['window']


        self.conv_1 = Conv2d(1,conv_1_filters,
                            kernel_size=(conv_1_window,emb_size))

        conv_2_filters = params['cnn2']['conv_size']
        conv_2_window = params['cnn2']['conv_size']
        self.conv_2 = Conv2d(conv_1_filters,conv_2_filters,
                            kernel_size=(conv_2_window,1))

        self.pool = lambda t: self.horizontal_pooling(t)


    def forward(self,inp):
        q,a = inp

        q = self.embs(q)
        a = self.embs(a)

        q = q.unsqueeze(1)
        a = a.unsqueeze(1)
        qemb = self.conv_1(q)
        aemb = self.conv_1(a)

        print(qemb.shape)
        print(aemb.shape)

        qemb = self.conv_2(qemb)
        aemb = self.conv_2(aemb)

        qemb = self.pool(qemb)
        aemb = self.pool(aemb)

        out = f.cosine_similarity(qemb,aemb)
        return out

    def horizontal_pooling(self,t):
        return f.max_pool1d(t,t.size(2)).view(t.size(0),-1)