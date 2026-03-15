import argparse
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np


def tokenize(text: str) -> list[str]:
    text = text.lower()
    # Keep only letters for a tiny baseline tokenizer.
    text = re.sub(r"[^a-z]+", " ", text)
    return text.split()


def build_vocab(tokens: list[str], min_count: int) -> tuple[dict[str, int], list[str], np.ndarray]:
    counts = Counter(tokens)
    # Drop rare words to keep the vocab small.
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort()
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = vocab
    freqs = np.array([counts[w] for w in vocab], dtype=np.float64)
    return word_to_id, id_to_word, freqs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def train_cbow(
    corpus_ids: list[int],
    vocab_size: int,
    embedding_dim: int,
    window_size: int,
    neg_samples: int,
    lr: float,
    epochs: int,
    patience: int,
    min_delta: float,
    neg_probs: np.ndarray,
    rng: np.random.Generator,
    report_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Small random init.
    scale = 0.5 / embedding_dim
    w_in = rng.uniform(-scale, scale, size=(vocab_size, embedding_dim))
    w_out = rng.uniform(-scale, scale, size=(vocab_size, embedding_dim))

    context_size = 2 * window_size
    total_loss = 0.0
    total_steps = 0
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_sum = 0.0
        step = 0
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for i in range(window_size, len(corpus_ids) - window_size):
            target_id = corpus_ids[i]
            context_ids = (
                corpus_ids[i - window_size : i] + corpus_ids[i + 1 : i + window_size + 1]
            )
            if not context_ids:
                continue
            # CBOW context mean.
            h = w_in[context_ids].mean(axis=0)

            # Positive term.
            score_pos = np.dot(w_out[target_id], h)
            sig_pos = sigmoid(score_pos)
            loss = -np.log(sig_pos + 1e-10)

            grad_out_pos = (sig_pos - 1.0) * h
            grad_h = (sig_pos - 1.0) * w_out[target_id]

            # Negative samples.
            neg_ids = rng.choice(vocab_size, size=neg_samples, replace=True, p=neg_probs)
            neg_vecs = w_out[neg_ids]
            scores_neg = neg_vecs @ h
            sig_neg = sigmoid(scores_neg)
            loss += -np.sum(np.log(sigmoid(-scores_neg) + 1e-10))

            grad_out_neg = sig_neg[:, None] * h[None, :]
            grad_h += np.sum(sig_neg[:, None] * neg_vecs, axis=0)

            # Update output vectors.
            w_out[target_id] -= lr * grad_out_pos
            np.add.at(w_out, neg_ids, -lr * grad_out_neg)

            # Spread gradient across context words.
            grad_in = grad_h / context_size
            np.add.at(w_in, context_ids, -lr * grad_in)

            loss_sum += loss
            total_loss += loss
            step += 1
            total_steps += 1
            epoch_loss_sum += loss
            epoch_steps += 1

            if report_every and step % report_every == 0:
                avg_loss = loss_sum / report_every
                acc_loss = total_loss / max(total_steps, 1)
                elapsed = time.time() - start_time
                print(
                    f"epoch {epoch} step {step} avg_loss {avg_loss:.4f} "
                    f"acc_loss {acc_loss:.4f} "
                    f"speed {report_every / max(elapsed, 1e-6):.1f} steps/s"
                )
                loss_sum = 0.0
                start_time = time.time()

        epoch_avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        acc_loss = total_loss / max(total_steps, 1)
        if report_every == 0:
            print(f"epoch {epoch} avg_loss {epoch_avg_loss:.4f} acc_loss {acc_loss:.4f}")
        else:
            print(f"epoch {epoch} done avg_loss {epoch_avg_loss:.4f} acc_loss {acc_loss:.4f}")

        if epoch_avg_loss + min_delta < best_loss:
            best_loss = epoch_avg_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    return w_in, w_out


def most_similar(
    word: str, word_to_id: dict[str, int], id_to_word: list[str], w_in: np.ndarray, topn: int
) -> list[tuple[str, float]]:
    if word not in word_to_id:
        return []
    idx = word_to_id[word]
    v = w_in[idx]
    # Cosine similarity.
    norms = np.linalg.norm(w_in, axis=1)
    v_norm = np.linalg.norm(v) + 1e-10
    sims = (w_in @ v) / (norms * v_norm + 1e-10)
    best = np.argsort(-sims)
    results = []
    for i in best:
        if i == idx:
            continue
        results.append((id_to_word[i], float(sims[i])))
        if len(results) >= topn:
            break
    return results


def least_similar(
    word: str, word_to_id: dict[str, int], id_to_word: list[str], w_in: np.ndarray, topn: int
) -> list[tuple[str, float]]:
    if word not in word_to_id:
        return []
    idx = word_to_id[word]
    v = w_in[idx]
    # Cosine similarity.
    norms = np.linalg.norm(w_in, axis=1)
    v_norm = np.linalg.norm(v) + 1e-10
    sims = (w_in @ v) / (norms * v_norm + 1e-10)
    worst = np.argsort(sims)
    results = []
    for i in worst:
        if i == idx:
            continue
        results.append((id_to_word[i], float(sims[i])))
        if len(results) >= topn:
            break
    return results


def save_vectors(path: Path, id_to_word: list[str], w_in: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(id_to_word)} {w_in.shape[1]}\n")
        for i, word in enumerate(id_to_word):
            vec = " ".join(f"{x:.6f}" for x in w_in[i])
            f.write(f"{word} {vec}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CBOW word2vec with negative sampling in NumPy")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset") / "The White Fang by Jack London.txt",
    )
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--neg_samples", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--report_every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--save_path", type=Path, default=Path("w2v_cbow.txt"))
    args = parser.parse_args()

    text = args.data.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenize(text)
    if args.max_tokens > 0:
        tokens = tokens[: args.max_tokens]

    word_to_id, id_to_word, freqs = build_vocab(tokens, args.min_count)
    tokens = [t for t in tokens if t in word_to_id]
    corpus_ids = [word_to_id[t] for t in tokens]
    vocab_size = len(id_to_word)

    print(f"tokens {len(tokens)} vocab {vocab_size}")

    # Negative sampling distribution.
    pow_freqs = freqs ** 0.75
    neg_probs = pow_freqs / pow_freqs.sum()

    rng = np.random.default_rng(args.seed)
    w_in, w_out = train_cbow(
        corpus_ids=corpus_ids,
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        neg_samples=args.neg_samples,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        neg_probs=neg_probs,
        rng=rng,
        report_every=args.report_every,
    )

    save_vectors(args.save_path, id_to_word, w_in)
    print(f"saved vectors to {args.save_path}")

    probe_words = ["wolf", "dog", "man", "fire", "snow", "love"]
    print("")
    print("most similar")
    for w in probe_words:
        sims = most_similar(w, word_to_id, id_to_word, w_in, topn=5)
        if not sims:
            continue
        sim_text = ", ".join(f"{sw}:{score:.3f}" for sw, score in sims)
        print(f"{w} -> {sim_text}")
    print("")
    print("least similar")
    for w in probe_words:
        sims = least_similar(w, word_to_id, id_to_word, w_in, topn=5)
        if not sims:
            continue
        sim_text = ", ".join(f"{sw}:{score:.3f}" for sw, score in sims)
        print(f"{w} -> {sim_text}")


if __name__ == "__main__":
    main()
