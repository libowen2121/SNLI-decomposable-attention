import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class encoder(nn.Module):

    def __init__(self, num_embeddings, embedding_size, hidden_size, para_init):
        super(encoder, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.para_init = para_init

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.input_linear1 = nn.Linear(
            self.embedding_size, self.hidden_size, bias=False)  # linear transformation
        self.input_linear2 = nn.Linear(
            self.embedding_size, self.hidden_size, bias=False)  # linear transformation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                # m.bias.data.uniform_(-0.01, 0.01)

    def forward(self, sent1, sent2):
        '''
               sent: batch_size x length (Long tensor)
        '''
        batch_size = sent1.size(0)
        sent1 = self.embedding(sent1)
        sent2 = self.embedding(sent2)

        sent1 = sent1.view(-1, self.embedding_size)
        sent2 = sent2.view(-1, self.embedding_size)

        sent1_linear = self.input_linear1(sent1).view(
            batch_size, -1, self.hidden_size)
        sent2_linear = self.input_linear2(sent2).view(
            batch_size, -1, self.hidden_size)

        return sent1_linear, sent2_linear



class embedding_linear(nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super(embedding_linear, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_linear = nn.Linear(
            self.embedding_size, self.hidden_size, bias=False)  # linear transformation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0., 0.01)
                # m.bias.data.zero_()

    def forward(self, sent1, sent2):
        '''
               sent: batch_size x length x embedding_size
        '''

        batch_size = sent1.size(0)

        # print sent1.size()

        sent1 = sent1.view(-1, self.embedding_size)
        sent2 = sent2.view(-1, self.embedding_size)

        # print sent1.size()

        # print self.input_linear(sent1).size()

        sent1_linear = self.input_linear(sent1).view(
            batch_size, -1, self.hidden_size)
        sent2_linear = self.input_linear(sent2).view(
            batch_size, -1, self.hidden_size)

        return sent1_linear, sent2_linear


class atten(nn.Module):
    '''
        intra sentence attention
    '''

    def __init__(self, hidden_size, label_size, para_init):
        super(atten, self).__init__()

        self.hidden_size = hidden_size
        self.label_size = label_size
        self.para_init = para_init

        self.mlp_f1 = self._mlp_layers(self.hidden_size, self.hidden_size)
        self.mlp_f2 = self._mlp_layers(self.hidden_size, self.hidden_size)
        self.mlp_g1 = self._mlp_layers(2 * self.hidden_size, self.hidden_size)
        self.mlp_g2 = self._mlp_layers(2 * self.hidden_size, self.hidden_size)
        self.mlp_h = self._mlp_layers(2 * self.hidden_size, self.hidden_size)

        self.final_linear = nn.Linear(
            self.hidden_size, self.label_size, bias=True)

        self.log_prob = nn.LogSoftmax()

        '''initialize parameters'''
        for m in self.modules():
            # print m
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                m.bias.data.normal_(0, self.para_init)

    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())        
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent1_linear, sent2_linear):
        '''
            sent_linear: batch_size x length x hidden_size
        '''
        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)

        '''attend'''

        f1 = self.mlp_f1(sent1_linear.view(-1, self.hidden_size))
        f2 = self.mlp_f2(sent2_linear.view(-1, self.hidden_size))

        f1 = f1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2)).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1)).view(-1, len2, len1)
        # batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (sent2_linear, torch.bmm(prob2, sent1_linear)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''sum'''
        g1 = self.mlp_g1(sent1_combine.view(-1, 2 * self.hidden_size))
        g2 = self.mlp_g2(sent2_combine.view(-1, 2 * self.hidden_size))
        g1 = g1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * hidden_size

        # if sample_id == 15:
        #     print '-2 layer'
        #     print h.data[:, 100:150]

        h = self.final_linear(h)

        # print 'final layer'
        # print h.data

        log_prob = self.log_prob(h)

        return log_prob
