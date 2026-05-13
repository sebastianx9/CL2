import argparse
import torch

from data import get_dataloaders, decode_ids, VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX
from model import EncoderDecoder


def greedy_generate(model, src_ids, max_len=64, device='cpu'):
    model.eval()
    if src_ids.dim() == 1:
        src_ids = src_ids.unsqueeze(0)
    src_ids = src_ids.to(device)

    with torch.no_grad():
        encoder_output, src_padding_mask = model.encode(src_ids)
        # start from BOS and generate one token at a time until EOS or max_len
        tgt = torch.tensor([[BOS_IDX]], device=device)

        for _ in range(max_len):
            decoder_out = model.decode(tgt, encoder_output, src_padding_mask)
            # take logits at the last position to predict the next token
            next_token = model.lm_head(decoder_out)[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == EOS_IDX:
                break

    return tgt.squeeze(0).tolist()


def beam_search(model, src_ids, beam_size=4, max_len=64, device='cpu'):
    model.eval()
    if src_ids.dim() == 1:
        src_ids = src_ids.unsqueeze(0)
    src_ids = src_ids.to(device)

    with torch.no_grad():
        encoder_output, src_padding_mask = model.encode(src_ids)
        # candidates: list of (cumulative_log_prob, sequence_tensor)
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


def evaluate(model, loader, device, decode='greedy', beam_size=4):
    model.eval()
    correct, total, total_ed = 0, 0, 0
    results = []

    with torch.no_grad():
        for src_ids, tgt_ids, lemmas, feats, forms in loader:
            for i in range(len(forms)):
                src = src_ids[i]
                # strip padding before passing to encoder
                src_trimmed = src[:(src != PAD_IDX).sum()]
                if decode == 'beam':
                    pred_ids = beam_search(model, src_trimmed, beam_size=beam_size, device=device)
                else:
                    pred_ids = greedy_generate(model, src_trimmed, device=device)
                pred = decode_ids(pred_ids)
                gold = forms[i]
                ed = edit_distance(gold, pred)
                if pred == gold:
                    correct += 1
                total    += 1
                total_ed += ed
                results.append((lemmas[i], feats[i], gold, pred, ed))

    acc    = correct / total if total else 0
    avg_ed = total_ed / total if total else 0
    return acc, avg_ed, results


def main():
    parser = argparse.ArgumentParser(description='Evaluate morphological inflection model')
    parser.add_argument('--checkpoint',     type=str,   required=True)
    parser.add_argument('--split',          type=str,   default='dev', choices=['dev', 'test'])
    parser.add_argument('--output',         type=str,   default='predictions.tsv')
    parser.add_argument('--decode',         type=str,   default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_size',      type=int,   default=4)
    parser.add_argument('--data_dir',       type=str,   default='.')
    parser.add_argument('--input_dim',      type=int,   default=128)
    parser.add_argument('--q_dim',          type=int,   default=64)
    parser.add_argument('--mlp_hidden_dim', type=int,   default=256)
    parser.add_argument('--n_enc_layers',   type=int,   default=3)
    parser.add_argument('--n_dec_layers',   type=int,   default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EncoderDecoder(
        vocab_size     = VOCAB_SIZE,
        input_dim      = args.input_dim,
        q_dim          = args.q_dim,
        mlp_hidden_dim = args.mlp_hidden_dim,
        n_enc_layers   = args.n_enc_layers,
        n_dec_layers   = args.n_dec_layers,
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint from {args.checkpoint}")

    _, dev_loader, test_loader = get_dataloaders(args.data_dir, batch_size=1)
    loader = dev_loader if args.split == 'dev' else test_loader

    acc, avg_ed, results = evaluate(model, loader, device, decode=args.decode, beam_size=args.beam_size)
    print(f"{args.split} [{args.decode}] | exact match: {acc:.4f} | avg edit distance: {avg_ed:.4f}")

    # save predictions
    with open(args.output, 'w', encoding='utf-8') as f:
        for lemma, feat, gold, pred, ed in results:
            f.write(f"{lemma}\t{feat}\t{pred}\n")
    print(f"Predictions saved to {args.output}")

    # print hardest cases
    errors = [(l, f, g, p, ed) for l, f, g, p, ed in results if ed > 0]
    errors.sort(key=lambda x: -x[4])
    print("\n--- Hardest cases (highest edit distance) ---")
    for lemma, feat, gold, pred, ed in errors[:10]:
        print(f"  ED={ed}  {lemma} + {feat}")
        print(f"    gold: {gold!r}")
        print(f"    pred: {pred!r}")


if __name__ == '__main__':
    main()
