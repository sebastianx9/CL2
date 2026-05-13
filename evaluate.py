import sys
import torch
from collections import defaultdict

from data import get_dataloaders, decode_ids, VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX
from model import EncoderDecoder

# --- Settings ---
BEAM_SIZE  = 4
N_HEADS    = 4
N_ENC_LAYERS = 2
N_DEC_LAYERS = 2

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
CHECKPOINT     = f'best_model_{_SIZE}.pth'
OUTPUT         = f'predictions_{_SIZE}.tsv'


def beam_search(model, src_ids, beam_size=4, max_len=64, device='cpu'):
    model.eval()
    if src_ids.dim() == 1:
        src_ids = src_ids.unsqueeze(0)
    src_ids = src_ids.to(device)

    with torch.no_grad():
        encoder_output, src_padding_mask = model.encode(src_ids)
        candidates = [(0.0, torch.tensor([[BOS_IDX]], device=device))]
        finished   = []

        for _ in range(max_len):
            expanded = []
            for score, seq in candidates:
                decoder_out = model.decode(seq, encoder_output, src_padding_mask)
                log_probs   = torch.log_softmax(model.lm_head(decoder_out)[:, -1, :], dim=-1)
                top_lp, top_ids = log_probs.topk(beam_size, dim=-1)
                for lp, idx in zip(top_lp[0].tolist(), top_ids[0].tolist()):
                    new_seq = torch.cat([seq, torch.tensor([[idx]], device=device)], dim=1)
                    expanded.append((score + lp, new_seq))

            expanded.sort(key=lambda x: x[0], reverse=True)
            candidates = []
            for score, seq in expanded:
                if seq[0, -1].item() == EOS_IDX:
                    finished.append((score, seq))
                else:
                    candidates.append((score, seq))
                if len(candidates) == beam_size:
                    break

            if len(finished) >= beam_size or not candidates:
                break

        if not finished:
            finished = candidates
        finished.sort(key=lambda x: x[0], reverse=True)
        return finished[0][1].squeeze(0).tolist()


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]


def evaluate(model, loader, device):
    model.eval()
    correct, total, total_ed = 0, 0, 0
    results = []

    with torch.no_grad():
        for src_ids, tgt_ids, lemmas, feats, forms in loader:
            for i in range(len(forms)):
                src = src_ids[i]
                src_trimmed = src[:(src != PAD_IDX).sum()]
                pred_ids = beam_search(model, src_trimmed, beam_size=BEAM_SIZE, device=device)
                pred = decode_ids(pred_ids)
                gold = forms[i]
                ed  = edit_distance(gold, pred)
                if pred == gold:
                    correct += 1
                total    += 1
                total_ed += ed
                results.append((lemmas[i], feats[i], gold, pred, ed))

    acc    = correct / total if total else 0
    avg_ed = total_ed / total if total else 0
    return acc, avg_ed, results


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EncoderDecoder(
        vocab_size     = VOCAB_SIZE,
        input_dim      = INPUT_DIM,
        q_dim          = Q_DIM,
        n_heads        = N_HEADS,
        mlp_hidden_dim = MLP_HIDDEN_DIM,
        n_enc_layers   = N_ENC_LAYERS,
        n_dec_layers   = N_DEC_LAYERS,
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"Loaded checkpoint: {CHECKPOINT}")

    _, _, test_loader = get_dataloaders(batch_size=1)

    acc, avg_ed, results = evaluate(model, test_loader, device)
    print(f"test | exact match: {acc:.4f} | avg edit distance: {avg_ed:.4f}")

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for lemma, feat, gold, pred, ed in results:
            f.write(f"{lemma}\t{feat}\t{pred}\n")
    print(f"Predictions saved to {OUTPUT}")

    errors = [(l, f, g, p, ed) for l, f, g, p, ed in results if ed > 0]
    errors.sort(key=lambda x: -x[4])
    print("\n--- Hardest cases (highest edit distance) ---")
    for lemma, feat, gold, pred, ed in errors[:10]:
        print(f"  ED={ed}  {lemma} + {feat}")
        print(f"    gold: {gold!r}")
        print(f"    pred: {pred!r}")

    print("\n--- Accuracy by feature tag ---")
    tag_correct = defaultdict(int)
    tag_total   = defaultdict(int)
    for _, feat, gold, pred, _ in results:
        tag_correct[feat] += int(gold == pred)
        tag_total[feat]   += 1
    for feat in sorted(tag_total, key=lambda t: -tag_total[t]):
        n = tag_total[feat]
        c = tag_correct[feat]
        print(f"  {feat}: {c}/{n} = {c/n:.3f}")
