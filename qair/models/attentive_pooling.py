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
                              windows=params['qcnn']['windows'],
                              activation=activations[params['qcnn']['activation']])

        self.conv_a = KimConv(params['emb_dim'],
                            params['acnn']['conv_size'],
                            windows=params['acnn']['windows'],
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

        self.h_pool = lambda x: self.horidef flatten(self,x):
    return x.view(x.size(0),-1)zontal_pooling(x)
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

