import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import Conv2d

from qair.models import Model
from qair.models.layers import KimConv, activations, AttentionMatrix, SimpleConv, BiLSTM

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

@Model.register("test-att-cnn")
class AP_CNN(Model):
  
    def __init__(self, params, vocab, device='cpu'):
        super().__init__()
        self.vocab = vocab

        self.device = torch.device(device)
        params['emb_num'], params['emb_dim'] = vocab.shape

        self.embs = nn.Embedding(params['emb_num'],
                                 params['emb_dim'],
                                 vocab['PAD'])

        try:
            self.shared = params['shared']
        except KeyError:
            self.shared = True


        self.convolution_q = SimpleConv(params['emb_dim'],params['qcnn']['conv_size'],params['qcnn']['window'])
        if self.shared:
            self.convolution_a = self.convolution_q
        else:
            self.convolution_a = SimpleConv(params['emb_dim'],params['acnn']['conv_size'],params['acnn']['window'])

        self.attention_mat = AttentionMatrix(params['qcnn']['conv_size'])
        self.h_pool = lambda t : self.horizontal_pooling(t)
        self.v_pool = lambda t : self.vertical_pooling(t)
        self.dense = nn.Sequential(
            nn.Linear(2*params['qcnn']['conv_size'], params['hidden_size']),
            nn.Tanh(),
            nn.Linear(params['hidden_size'], 1)
        )
    
    def flatten(self,x):
      return x.view(x.size(0),-1)
    
    def forward(self,inp):
        q,a = inp
        q_emb = self.embs(q)
        a_emb = self.embs(a)
        # q_enc = self.convolution(q_emb)
        # a_enc = self.convolution(a_emb)
        q_enc = self.convolution_q(q_emb)
        a_enc = self.convolution_a(a_emb)
        mat = self.attention_mat(q_enc,a_enc)
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



@Model.register("test-att-lstm")
class AP_LSTM(Model):
  
    def __init__(self, params, vocab, device='cpu'):
        super().__init__()
        self.vocab = vocab

        self.device = torch.device(device)
        params['emb_num'], params['emb_dim'] = vocab.shape

        self.embs = nn.Embedding(params['emb_num'],
                                 params['emb_dim'],
                                 vocab['PAD'])

        try:
            act = activations[params['lstm']['activation']]
        except KeyError:
            act = None

        try:
            self.shared = params['shared']
        except KeyError:
            self.shared = True

        lstm_hidden = params['lstm']['single_hidden_dim'] * 2
        self.lstm_q = BiLSTM(params['emb_dim'],params['lstm']['single_hidden_dim'],activation=act)
        if self.shared:
            self.lstm_a = self.lstm_q
        else:
            self.lstm_a = BiLSTM(params['emb_dim'],params['lstm']['single_hidden_dim'],activation=act)

        self.attention_mat = AttentionMatrix(lstm_hidden)
        self.h_pool = lambda t : self.horizontal_pooling(t)
        self.v_pool = lambda t : self.vertical_pooling(t)
        self.dense = nn.Sequential(
            nn.Linear(2*lstm_hidden, params['hidden_size']),
            nn.Tanh(),
            nn.Linear(params['hidden_size'], 1),
        )
    
    def flatten(self,x):
      return x.view(x.size(0),-1)
    
    def forward(self,inp):
        q,a = inp
        q_emb = self.embs(q)
        a_emb = self.embs(a)
        q_enc = self.lstm_q(q_emb).transpose(1,2)
        a_enc = self.lstm_a(a_emb).transpose(1,2)
        mat = self.attention_mat(q_enc,a_enc)
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
