import torch
import torch.nn as nn
import torch.nn.functional as f

from qair.models import Model
from qair.models.layers import KimConv, activations

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

    def horizontal_pooling(self,t):
    	print(t.shape)
        return f.max_pool1d(t,t.size(2)).view(t.size(0),-1)

    def forward(self, inp):
        (q, a) = inp
        # Map ids to word embeddings
        q = self.embs(q)
        a = self.embs(a)

        # Convolutional Encoder

        qemb = self.conv_q(q)
        aemb = self.conv_a(a)

        qout = self.horizontal_pooling(qemb)
        aout = self.horizontal_pooling(aemb)

       	out = f.cosine_similarity(qout,aout)

        return out

