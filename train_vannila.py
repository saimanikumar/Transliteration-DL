# train.py
import os
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DakshinaTSVDataset
from model import Encoder, Decoder, Seq2Seq

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train vanilla Seq2Seq transliteration model"
    )
    # data & vocab
    parser.add_argument("--train_tsv",   type=str, required=True, help="Path to train .tsv")
    parser.add_argument("--dev_tsv",     type=str, required=True, help="Path to dev .tsv")
    parser.add_argument("--src_vocab",   type=str, required=True, help="Path to src_vocab.json")
    parser.add_argument("--tgt_vocab",   type=str, required=True, help="Path to tgt_vocab.json")
    parser.add_argument("--model_dir",   type=str, default="models", help="Where to save checkpoints")

    # model hyperparameters
    parser.add_argument("--emb_size",    type=int,   default=256,   help="Embedding dimension")
    parser.add_argument("--hid_size",    type=int,   default=512,   help="Hidden state dimension")
    parser.add_argument("--num_layers",  type=int,   default=2,     help="Number of RNN layers")
    parser.add_argument("--dropout",     type=float, default=0.1,   help="Dropout between layers")
    parser.add_argument("--cell_type",   type=str,   default="gru", help="rnn | gru | lstm")
    parser.add_argument("--bidirectional", action="store_true",     help="Use bidirectional encoder")

    # training hyperparameters
    parser.add_argument("--batch_size",       type=int,   default=64,     help="Batch size")
    parser.add_argument("--lr",               type=float, default=1e-3,   help="Learning rate")
    parser.add_argument("--epochs",           type=int,   default=15,     help="Number of epochs")
    parser.add_argument("--teacher_forcing",  type=float, default=0.7,    help="Teacher forcing ratio")
    parser.add_argument("--clip_grad",        type=float, default=1.0,    help="Max grad norm clipping")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # load vocab
    with open(args.src_vocab, 'r', encoding='utf-8') as f:
        src_vocab = json.load(f)
    with open(args.tgt_vocab, 'r', encoding='utf-8') as f:
        tgt_vocab = json.load(f)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets & loaders
    train_ds = DakshinaTSVDataset(args.train_tsv, src_vocab, tgt_vocab)
    dev_ds   = DakshinaTSVDataset(args.dev_tsv,   src_vocab, tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size)

    # model
    encoder = Encoder(
        input_size=len(src_vocab),
        emb_size=args.emb_size,
        hid_size=args.hid_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        cell_type=args.cell_type
    )
    decoder = Decoder(
        output_size=len(tgt_vocab),
        emb_size=args.emb_size,
        hid_size=(args.hid_size * (2 if args.bidirectional else 1)),
        num_layers=args.num_layers,
        dropout=args.dropout,
        cell_type=args.cell_type
    )
    model = Seq2Seq(encoder, decoder, device, teacher_forcing=args.teacher_forcing)
    model = model.to(device)

    # optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_dev_acc = 0.0
    for ep in range(1, args.epochs + 1):
        # training
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(src, tgt)
            loss = criterion(out.view(-1, out.size(-1)), tgt[:,1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for src, tgt in dev_loader:
                src, tgt = src.to(device), tgt.to(device)
                out = model(src, tgt)
                preds = out.argmax(dim=-1)
                for i in range(src.size(0)):
                    # convert to strings and compare
                    pred_tokens = [
                        list(tgt_vocab.keys())[list(tgt_vocab.values()).index(pid.item())]
                        for pid in preds[i] if pid.item() not in {0,1,2,3}
                    ]
                    tgt_tokens = []
                    for tid in tgt[i,1:]:
                        if tid.item() in {0,2}: break
                        if tid.item() not in {1,3}:
                            tgt_tokens.append(
                                list(tgt_vocab.keys())[list(tgt_vocab.values()).index(tid.item())]
                            )
                    if "".join(pred_tokens) == "".join(tgt_tokens):
                        correct += 1
                    total += 1
        dev_acc = correct / total

        print(f"Epoch {ep}: Train Loss={avg_train_loss:.4f}, Dev Acc={dev_acc:.4f}")

        # checkpoint
        ckpt_path = os.path.join(args.model_dir, f"vanilla_ep{ep}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # best
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best_vanilla.pt"))

    print(f"Training complete. Best dev exact-match accuracy: {best_dev_acc:.4f}")

if __name__ == "__main__":
    main()
