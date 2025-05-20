# model_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1,
                 dropout=0.0, bidirectional=True, cell_type='lstm'):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.bidirectional = bidirectional
        self.cell_type    = cell_type.lower()

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)

        rnn_cls = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[self.cell_type]
        self.rnn = rnn_cls(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers>1 else 0.0),
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, src):
        # src: [B, T]
        mask = (src != 0).float()                 # [B, T]
        emb  = self.embedding(src)                # [B, T, E]
        outputs, hidden = self.rnn(emb)           # outputs: [B, T, H*dir]

        # If bidirectional, collapse layers*2 → layers with concatenated hidden
        if self.bidirectional:
            if isinstance(hidden, tuple):  # LSTM
                h, c = hidden
                h = h.view(self.num_layers, 2, -1, self.hidden_size)
                c = c.view(self.num_layers, 2, -1, self.hidden_size)
                h = torch.cat([h[:,0], h[:,1]], dim=2)
                c = torch.cat([c[:,0], c[:,1]], dim=2)
                hidden = (h, c)
            else:  # GRU/RNN
                h = hidden.view(self.num_layers, 2, -1, self.hidden_size)
                hidden = torch.cat([h[:,0], h[:,1]], dim=2)

        return outputs, hidden, mask


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.energy = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v      = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, dec_h, enc_outs, mask):
        # dec_h: [B, D], enc_outs: [B, T, E], mask: [B, T]
        B, T, _ = enc_outs.size()
        dec_rep = dec_h.unsqueeze(1).repeat(1, T, 1)           # [B, T, D]
        e = torch.tanh(self.energy(torch.cat([dec_rep, enc_outs], dim=2)))  # [B, T, D]
        scores = self.v(e).squeeze(2)                          # [B, T]
        scores = scores.masked_fill(mask==0, -1e10)
        return F.softmax(scores, dim=1)                        # [B, T]


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, enc_hidden_size,
                 dec_hidden_size, num_layers=1, dropout=0.0, cell_type='lstm'):
        super().__init__()
        self.output_size    = output_size
        self.dec_hidden_size = dec_hidden_size
        self.cell_type      = cell_type.lower()

        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=0)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)

        rnn_input_dim = embedding_size + enc_hidden_size
        rnn_cls = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[self.cell_type]
        self.rnn = rnn_cls(
            rnn_input_dim, dec_hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers>1 else 0.0),
            batch_first=True
        )

        self.fc_out = nn.Linear(dec_hidden_size + enc_hidden_size + embedding_size,
                                output_size)

    def forward(self, inp, hidden, enc_outs, mask):
        # inp: [B], hidden: ([layers,B,H],...) , enc_outs: [B, T, E]
        emb   = self.embedding(inp).unsqueeze(1)  # [B,1,Emb]
        if self.cell_type=='lstm':
            dec_h = hidden[0][-1]                  # [B,H]
        else:
            dec_h = hidden[-1]

        attn_w = self.attention(dec_h, enc_outs, mask)        # [B, T]
        context = torch.bmm(attn_w.unsqueeze(1), enc_outs)    # [B,1, E]

        rnn_in = torch.cat([emb, context], dim=2)             # [B,1, Emb+E]
        out, hidden = self.rnn(rnn_in, hidden)                # out: [B,1,H]

        out, context, emb = out.squeeze(1), context.squeeze(1), emb.squeeze(1)
        comb = torch.cat([out, context, emb], dim=1)          # [B, H+E+Emb]
        pred = self.fc_out(comb)                              # [B, V]
        return pred, hidden, attn_w


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.7):
        super().__init__()
        self.encoder               = encoder
        self.decoder               = decoder
        self.device                = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, tgt):
        B, T = src.size()
        V = self.decoder.output_size
        outputs = torch.zeros(B, T-1, V, device=self.device)

        enc_outs, hidden, mask = self.encoder(src)
        inp = tgt[:, 0]  # <sos>

        for t in range(1, T):
            pred, hidden, _ = self.decoder(inp, hidden, enc_outs, mask)
            outputs[:, t-1] = pred
            if random.random() < self.teacher_forcing_ratio:
                inp = tgt[:, t]
            else:
                inp = pred.argmax(1)

        return outputs

    @torch.no_grad()
    def decode(self, src, max_len=100):
        B, _ = src.size()
        enc_outs, hidden, mask = self.encoder(src)
        inp = torch.full((B,), 3, dtype=torch.long, device=self.device)  # <sos>=3
        outs, atts = [], []

        for _ in range(max_len):
            pred, hidden, attn_w = self.decoder(inp, hidden, enc_outs, mask)
            inp = pred.argmax(1)
            outs.append(inp)
            atts.append(attn_w)
            if (inp==2).all():  # <eos>=2
                break

        outputs    = torch.stack(outs, dim=1)   # [B, L]
        attentions = torch.stack(atts, dim=1)   # [B, L, T]
        return outputs, attentions
