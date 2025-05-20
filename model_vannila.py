# model.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hid_size, num_layers=1, dropout=0.0,
                 cell_type='gru', bidirectional=False):
        super().__init__()
        self.emb = nn.Embedding(input_size, emb_size, padding_idx=0)
        rnn_cls = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=emb_size,
            hidden_size=hid_size,
            num_layers=num_layers,
            dropout=dropout if num_layers>1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
    def forward(self, src):
        # src: [B, T]
        emb = self.emb(src)              # [B, T, emb_size]
        outputs, hidden = self.rnn(emb)  # hidden: [layers*dir, B, hid]
        # if bidirectional, user can reshape externally
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, hid_size, num_layers=1, dropout=0.0,
                 cell_type='gru'):
        super().__init__()
        self.emb = nn.Embedding(output_size, emb_size, padding_idx=0)
        rnn_cls = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=emb_size,
            hidden_size=hid_size,
            num_layers=num_layers,
            dropout=dropout if num_layers>1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hid_size, output_size)

    def forward(self, input_step, hidden):
        # input_step: [B]  single token ids
        # hidden: [layers, B, hid]
        emb = self.emb(input_step).unsqueeze(1)  # [B,1,emb_size]
        out, hidden = self.rnn(emb, hidden)
        out = out.squeeze(1)                     # [B, hid]
        logits = self.fc(out)                    # [B, V]
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing = teacher_forcing

    def forward(self, src, tgt):
        B, T = src.size()
        V = self.decoder.fc.out_features

        outputs = torch.zeros(B, T-1, V).to(self.device)
        enc_out, hidden = self.encoder(src)

        input_step = tgt[:,0]
        for t in range(1, T):
            logits, hidden = self.decoder(input_step, hidden)
            outputs[:, t-1] = logits
            use_tf = torch.rand(1).item() < self.teacher_forcing
            input_step = tgt[:,t] if use_tf else logits.argmax(dim=1)
        return outputs


