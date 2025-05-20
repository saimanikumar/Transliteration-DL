import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random

# ---- Dataset Definition with special tokens ----
class DakshinaTSVDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file, src_vocab=None, tgt_vocab=None, max_len=64, build_vocab=False):
        df = pd.read_csv(tsv_file, sep='\t', header=None,
                         names=['native', 'roman', 'freq'], usecols=[0, 1], dtype=str)
        # Fix the pandas warning by using a copy
        df = df.copy()
        df['native'] = df['native'].fillna('')
        df['roman'] = df['roman'].fillna('')
        self.pairs = list(zip(df['roman'], df['native']))
        print(f"Loaded {len(self.pairs)} examples from {tsv_file}")
        
        # Print a few examples
        if len(self.pairs) > 0:
            print("Sample examples:")
            for i in range(min(3, len(self.pairs))):
                print(f"  Roman: '{self.pairs[i][0]}', Native: '{self.pairs[i][1]}'")
                
        self.max_len = max_len
        
        if build_vocab:
            self.src_vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3}
            self.tgt_vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3}
            self._build_vocab()
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
            # Ensure special tokens exist
            for v in ('<eos>', '<sos>'):
                if v not in self.src_vocab: self.src_vocab[v] = len(self.src_vocab)
                if v not in self.tgt_vocab: self.tgt_vocab[v] = len(self.tgt_vocab)

    def _build_vocab(self):
        for src, tgt in self.pairs:
            for ch in src:
                if ch not in self.src_vocab: self.src_vocab[ch] = len(self.src_vocab)
            for ch in tgt:
                if ch not in self.tgt_vocab: self.tgt_vocab[ch] = len(self.tgt_vocab)
        print(f"Vocab sizes -> src: {len(self.src_vocab)}, tgt: {len(self.tgt_vocab)}")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        
        # Add <sos> and <eos> tokens
        src_idxs = [self.src_vocab['<sos>']] + [self.src_vocab.get(ch, self.src_vocab['<unk>']) for ch in src] + [self.src_vocab['<eos>']]
        tgt_idxs = [self.tgt_vocab['<sos>']] + [self.tgt_vocab.get(ch, self.tgt_vocab['<unk>']) for ch in tgt] + [self.tgt_vocab['<eos>']]
        
        # Pad sequences
        pad_src = [self.src_vocab['<pad>']] * max(0, self.max_len - len(src_idxs))
        pad_tgt = [self.tgt_vocab['<pad>']] * max(0, self.max_len - len(tgt_idxs))
        
        # Truncate if necessary and convert to tensor
        src_tensor = torch.tensor((src_idxs + pad_src)[:self.max_len], dtype=torch.long)
        tgt_tensor = torch.tensor((tgt_idxs + pad_tgt)[:self.max_len], dtype=torch.long)
        
        return src_tensor, tgt_tensor

# ---- Encoder with bidirectional support ----
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout=0, bidirectional=True, cell_type='lstm'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type.lower()
        
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_size, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                embedding_size, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:  # rnn
            self.rnn = nn.RNN(
                embedding_size, 
                hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
            
        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)
                
    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size = x.shape[0]
        
        # Create mask for attention
        mask = (x != 0).float()  # 0 is <pad>
        
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_size]
        
        # Pass through RNN
        outputs, hidden = self.rnn(embedded)
        
        # Process hidden state based on RNN type
        if self.cell_type == 'lstm':
            hidden_state, cell_state = hidden
            
            if self.bidirectional:
                # Reshape hidden from [num_layers*2, batch_size, hidden_size]
                # to [num_layers, 2, batch_size, hidden_size]
                hidden_state = hidden_state.view(self.num_layers, 2, batch_size, self.hidden_size)
                cell_state = cell_state.view(self.num_layers, 2, batch_size, self.hidden_size)
                
                # Concatenate bidirectional states
                hidden_state = torch.cat([hidden_state[:, 0], hidden_state[:, 1]], dim=2)
                cell_state = torch.cat([cell_state[:, 0], cell_state[:, 1]], dim=2)
                
                # Final hidden state is now [num_layers, batch_size, hidden_size*2]
                hidden = (hidden_state, cell_state)
            
        else:  # GRU or RNN
            if self.bidirectional:
                # Reshape hidden from [num_layers*2, batch_size, hidden_size]
                # to [num_layers, 2, batch_size, hidden_size]
                hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
                
                # Concatenate bidirectional states
                hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
                
                # Final hidden state is now [num_layers, batch_size, hidden_size*2]
        
        # For bidirectional, output is [batch_size, seq_len, hidden_size*2]
        return outputs, hidden, mask

# ---- Attention Mechanism ----
class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        # Create a linear layer to convert the concatenated hidden states to attention scores
        self.energy = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch_size, dec_hidden_size]
        # encoder_outputs: [batch_size, src_len, enc_hidden_size]
        # mask: [batch_size, src_len]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        # [batch_size, dec_hidden_size] -> [batch_size, src_len, dec_hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Create energy by concatenating encoder outputs and decoder hidden
        # [batch_size, src_len, enc_hidden_size + dec_hidden_size]
        energy = torch.cat((hidden, encoder_outputs), dim=2)
        
        # Apply attention layer
        # [batch_size, src_len, dec_hidden_size]
        energy = torch.tanh(self.energy(energy))
        
        # Get attention scores
        # [batch_size, src_len, 1]
        attention = self.v(energy)
        
        # [batch_size, src_len]
        attention = attention.squeeze(2)
        
        # Mask out padding positions
        attention = attention.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get probabilities
        # [batch_size, src_len]
        return F.softmax(attention, dim=1)

# ---- Decoder with attention and teacher forcing ----
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, enc_hidden_size, dec_hidden_size, 
                 num_layers, dropout=0, cell_type='lstm'):
        super().__init__()
        self.output_size = output_size
        self.dec_hidden_size = dec_hidden_size
        self.enc_hidden_size = enc_hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=0)
        
        # Initialize attention mechanism
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        
        # Context vector + embedding size as input to RNN
        rnn_input_size = embedding_size + enc_hidden_size
        
        # Initialize RNN based on cell type
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                rnn_input_size, 
                dec_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                rnn_input_size, 
                dec_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # rnn
            self.rnn = nn.RNN(
                rnn_input_size, 
                dec_hidden_size, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        # Final output layer that combines decoder output, context and embedding
        self.fc_out = nn.Linear(dec_hidden_size + enc_hidden_size + embedding_size, output_size)
        
        # Initialize weights using Xavier initialization
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)
                
    def forward(self, input, hidden, encoder_outputs, mask):
        # input: [batch_size]
        # hidden: [num_layers, batch_size, dec_hidden_size] or tuple for LSTM
        # encoder_outputs: [batch_size, src_len, enc_hidden_size]
        # mask: [batch_size, src_len]
        
        # Embed input token
        # [batch_size] -> [batch_size, 1, embedding_size]
        embedded = self.embedding(input).unsqueeze(1)
        
        # Get attention weights
        # [batch_size, src_len]
        if self.cell_type == 'lstm':
            h_for_attn = hidden[0][-1]  # use last layer's hidden state
        else:
            h_for_attn = hidden[-1]  # use last layer's hidden state
            
        attn_weights = self.attention(h_for_attn, encoder_outputs, mask)
        
        # Create context vector by weighting encoder outputs with attention
        # [batch_size, 1, src_len] * [batch_size, src_len, enc_hidden_size]
        # -> [batch_size, 1, enc_hidden_size]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        # Combine embedded token and context vector
        # [batch_size, 1, embedding_size + enc_hidden_size]
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # Pass through RNN
        # output: [batch_size, 1, dec_hidden_size]
        # hidden: [num_layers, batch_size, dec_hidden_size] or tuple for LSTM
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Combine output, context and embedding for final prediction
        # [batch_size, 1, dec_hidden_size + enc_hidden_size + embedding_size]
        output = torch.cat((output, context, embedded), dim=2)
        
        # Remove sequence dimension
        # [batch_size, dec_hidden_size + enc_hidden_size + embedding_size]
        output = output.squeeze(1)
        
        # Pass through final linear layer
        # [batch_size, output_size]
        prediction = self.fc_out(output)
        
        return prediction, hidden, attn_weights

# ---- Complete Seq2Seq Model with Teacher Forcing ----
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.7):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
    def forward(self, src, tgt):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_size
        
        # Tensor to store outputs
        outputs = torch.zeros(batch_size, tgt_len-1, tgt_vocab_size).to(self.device)
        
        # Encode source
        encoder_outputs, hidden, mask = self.encoder(src)
        
        # First input to decoder is the <sos> token (already embedded in tgt)
        input = tgt[:, 0]
        
        # Teacher forcing ratio determines how often to use true target as input
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        
        # Decode one token at a time
        for t in range(1, tgt_len):
            # Get output from decoder
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            
            # Store output
            outputs[:, t-1] = output
            
            # Next input is either true target (teacher forcing) or predicted token
            if use_teacher_forcing:
                input = tgt[:, t]
            else:
                # Get highest scoring token
                input = output.argmax(1)
                
        return outputs
    
    # For inference (no teacher forcing)
    def decode(self, src, max_len=100):
        # src: [batch_size, src_len]
        
        batch_size = src.shape[0]
        
        # Encode source
        encoder_outputs, hidden, mask = self.encoder(src)
        
        # First input is <sos> token
        input = torch.ones(batch_size, dtype=torch.long).to(self.device) * 3  # <sos> = 3
        
        # Track generated tokens
        outputs = [input]
        attentions = []
        
        # Track if sequence has ended
        ended = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        
        # Decode until max length or all sequences end
        for t in range(1, max_len):
            # Get output from decoder
            output, hidden, attn = self.decoder(input, hidden, encoder_outputs, mask)
            
            # Get next token
            input = output.argmax(1)
            
            # Store output
            outputs.append(input)
            attentions.append(attn)
            
            # Check if all sequences have ended
            ended = ended | (input == 2)  # 2 is <eos>
            if ended.all():
                break
                
        # Convert list of tensors to single tensor
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len]
        attentions = torch.stack(attentions, dim=1)  # [batch_size, seq_len-1, src_len]
        
        return outputs, attentions

# ---- Metrics & Utils ----
def compute_exact_match_accuracy(preds, targets, tgt_vocab):
    """Compute exact match accuracy between predictions and targets"""
    batch_size = preds.size(0)
    correct = 0
    
    # Convert ids to strings
    id_to_char = {v: k for k, v in tgt_vocab.items() if k not in ['<pad>', '<sos>', '<eos>', '<unk>']}
    
    for i in range(batch_size):
        # Extract character sequences (removing special tokens)
        pred_seq = ''.join([id_to_char.get(idx.item(), '') for idx in preds[i, 1:] 
                            if idx.item() not in [0, 1, 2, 3]])  # Skip <pad>, <unk>, <eos>, <sos>
        
        # For target, skip first token (<sos>) and stop at <eos> or <pad>
        tgt_seq = ''
        for idx in targets[i, 1:]:  # Skip first token
            token_id = idx.item()
            if token_id in [0, 2]:  # <pad> or <eos>
                break
            if token_id not in [1, 3]:  # Skip <unk> and <sos>
                tgt_seq += id_to_char.get(token_id, '')
        
        # Check for exact match
        if pred_seq == tgt_seq:
            correct += 1
    
    return correct / batch_size

def compute_char_accuracy(logits, targets):
    """Compute character-level accuracy between logits and targets"""
    preds = logits.argmax(dim=-1)
    mask = (targets != 0)  # Ignore padding
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0