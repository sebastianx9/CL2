"""
train.py — training entry point for the morphological inflection model.

Usage:
    python train.py
    python train.py --input_dim 256 --batch_size 32 --max_epochs 200
    python train.py --save_path checkpoints/my_model.pth

Data is downloaded automatically from SIGMORPHON 2023 on first run.
"""

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import get_dataloaders, VOCAB_SIZE, PAD_IDX
from model import EncoderDecoder


def compute_loss(logits, tgt_ids, pad_idx=PAD_IDX):
    # teacher forcing: given BOS t1...t_{n-1}, predict t1...tn EOS
    # shift logits and labels by one position to align predictions with targets
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    logits  = logits[:, :-1].reshape(-1, logits.size(-1))
    targets = tgt_ids[:, 1:].reshape(-1)
    return loss_fn(logits, targets)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for src_ids, tgt_ids, *_ in loader:
        src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
        optimizer.zero_grad()
        logits = model(src_ids, tgt_ids)
        loss   = compute_loss(logits, tgt_ids)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_epoch(model, loader, device):
    # validation also uses teacher forcing, so accuracy is optimistic vs. free generation
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for src_ids, tgt_ids, *_ in loader:
            src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
            logits = model(src_ids, tgt_ids)
            total_loss += compute_loss(logits, tgt_ids).item()

            preds   = logits[:, :-1].argmax(dim=-1)
            targets = tgt_ids[:, 1:]
            non_pad = targets != PAD_IDX
            total_correct += (preds[non_pad] == targets[non_pad]).sum().item()
            total_tokens  += non_pad.sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens else 0
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train morphological inflection model')
    parser.add_argument('--data_dir',       type=str,   default='.')
    parser.add_argument('--save_path',      type=str,   default='best_model.pth')
    parser.add_argument('--input_dim',      type=int,   default=128)
    parser.add_argument('--q_dim',          type=int,   default=64)
    parser.add_argument('--mlp_hidden_dim', type=int,   default=256)
    parser.add_argument('--n_enc_layers',   type=int,   default=3)
    parser.add_argument('--n_dec_layers',   type=int,   default=3)
    parser.add_argument('--dropout_p',      type=float, default=0.1)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--max_epochs',     type=int,   default=150)
    parser.add_argument('--patience',       type=int,   default=10)
    parser.add_argument('--seed',           type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, dev_loader, _ = get_dataloaders(args.data_dir, args.batch_size)

    model = EncoderDecoder(
        vocab_size     = VOCAB_SIZE,
        input_dim      = args.input_dim,
        q_dim          = args.q_dim,
        mlp_hidden_dim = args.mlp_hidden_dim,
        n_enc_layers   = args.n_enc_layers,
        n_dec_layers   = args.n_dec_layers,
        dropout_p      = args.dropout_p,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss    = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, args.max_epochs + 1):
        train_loss        = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate_epoch(model, dev_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_tok_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"           ^ saved best model to {args.save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses,   label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.title('Training dynamics')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_dynamics.png', dpi=150)
    print("Training curve saved to training_dynamics.png")


if __name__ == '__main__':
    main()
