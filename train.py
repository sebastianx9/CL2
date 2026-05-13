import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import get_dataloaders, VOCAB_SIZE, PAD_IDX
from model import EncoderDecoder


SEED       = 42
BATCH_SIZE = 64
N_HEADS    = 4
N_ENC_LAYERS = 2
N_DEC_LAYERS = 2
DROPOUT_P  = 0.3
LR         = 1e-3
MAX_EPOCHS = 150
PATIENCE   = 20

_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
_DIM_CONFIGS = {
    128: dict(input_dim=128, q_dim=32, mlp_hidden_dim=256),
    64:  dict(input_dim=64,  q_dim=16, mlp_hidden_dim=128),
    32:  dict(input_dim=32,  q_dim=8,  mlp_hidden_dim=64),
}
assert _SIZE in _DIM_CONFIGS, f"SIZE must be one of {list(_DIM_CONFIGS)}"
INPUT_DIM      = _DIM_CONFIGS[_SIZE]['input_dim']
Q_DIM          = _DIM_CONFIGS[_SIZE]['q_dim']
MLP_HIDDEN_DIM = _DIM_CONFIGS[_SIZE]['mlp_hidden_dim']
SAVE_PATH      = f'best_model_{_SIZE}.pth'


def compute_loss(logits, tgt_ids, pad_idx=PAD_IDX):
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
    # validation uses teacher forcing, so accuracy is optimistic vs. free generation
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


if __name__ == '__main__':
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Model size: {_SIZE}-dim")

    train_loader, dev_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    model = EncoderDecoder(
        vocab_size     = VOCAB_SIZE,
        input_dim      = INPUT_DIM,
        q_dim          = Q_DIM,
        n_heads        = N_HEADS,
        mlp_hidden_dim = MLP_HIDDEN_DIM,
        n_enc_layers   = N_ENC_LAYERS,
        n_dec_layers   = N_DEC_LAYERS,
        dropout_p      = DROPOUT_P,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss    = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss        = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate_epoch(model, dev_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_tok_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"           ^ saved best model to {SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
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
    plt.savefig(f'training_dynamics_{_SIZE}.png', dpi=150)
    print(f"Training curve saved to training_dynamics_{_SIZE}.png")
