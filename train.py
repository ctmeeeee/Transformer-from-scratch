from tqdm import tqdm
import torch
import torch.nn as nn
from model.transformer import Transformer
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy

'''
    Multi30k的数据集样本格式为:
        (en, de)
    的元组
'''
ds = load_dataset("bentrevett/multi30k")
#print(ds)

spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]


# ------ 构建词表 ------
def build_vocab(texts, tokenizer, min_freq=2):
    counter = Counter()
    for text in texts:
        tokens = tokenizer(text)
        counter.update(tokens)
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

# ------ 数字化文本 ------
def numericalize(tokens, vocab, bos=True, eos=True):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if bos: ids = [vocab["<bos>"]] + ids
    if eos: ids = ids + [vocab["<eos>"]]
    return ids

# ------ Padding ------
def pad_sequence(sequences, pad_idx):
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    pad_idx_src = de_vocab["<pad>"]
    pad_idx_tgt = en_vocab["<pad>"]

    src_batch = pad_sequence(src_batch, pad_idx_src)
    tgt_batch = pad_sequence(tgt_batch, pad_idx_tgt)

    src_batch = torch.tensor(src_batch, dtype=torch.long)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.long)

    return src_batch, tgt_batch

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab):
        self.src_data = [numericalize(tokenize_de(s), src_vocab) for s in src_texts]
        self.tgt_data = [numericalize(tokenize_en(t), tgt_vocab) for t in tgt_texts]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]
    
de_texts = [example['de'] for example in ds['train']]
en_texts = [example['en'] for example in ds['train']]

de_vocab = build_vocab(de_texts, tokenize_de)
en_vocab = build_vocab(en_texts, tokenize_en)
    
train_dataset = TranslationDataset(de_texts, en_texts, de_vocab, en_vocab)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

de_texts_val = [example['de'] for example in ds['validation']]
en_texts_val = [example['en'] for example in ds['validation']]

val_dataset = TranslationDataset(de_texts_val, en_texts_val, de_vocab, en_vocab)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)


# ------ 创建Mask -------

def create_padding_mask(seq, pad_idx):
    # seq_shape: (batch_size, seq_len)
    # return_size: (batch_size, 1, 1, seq_len)
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

def create_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# ------ 训练 ------

# ------ 创建Mask -------

def create_padding_mask(seq, pad_idx):
    # seq_shape: (batch_size, seq_len)
    # return_size: (batch_size, 1, 1, seq_len)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_subsequent_mask(size):
    mask = torch.tril(torch.ones((size, size), dtype=torch.bool))
    return mask

# ------ 训练 ------

model = Transformer(len(de_vocab), len(en_vocab))
criterion = nn.CrossEntropyLoss(ignore_index=en_vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def train_model(num_epochs, device, train_loader, val_loader=None):
    model.to(device)
    best_loss = float('inf')  # 初始最佳损失
    best_model_path = "best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0

        for src_batch, tgt_batch in progress:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            src_mask = create_padding_mask(src_batch, pad_idx=de_vocab["<pad>"])
            memory_mask = create_padding_mask(src_batch, pad_idx=de_vocab["<pad>"])

            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            batch_size, tgt_len = tgt_input.size()

            tgt_pad_mask = create_padding_mask(tgt_input, pad_idx=en_vocab["<pad>"])
            tgt_sub_mask = create_subsequent_mask(tgt_len).to(tgt_input.device)
            tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(1)
            tgt_mask = tgt_pad_mask & tgt_sub_mask

            enc_output = model.encoder(src_batch, src_mask)
            logits = model.decoder(enc_output, tgt_input, tgt_mask=tgt_mask, memory_mask=memory_mask)

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Training Loss: {total_loss / len(train_loader):.4f}")

        # ============ 验证逻辑 ============
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for src_val, tgt_val in val_loader:
                    src_val = src_val.to(device)
                    tgt_val = tgt_val.to(device)

                    src_mask = create_padding_mask(src_val, pad_idx=de_vocab["<pad>"])
                    memory_mask = create_padding_mask(src_val, pad_idx=de_vocab["<pad>"])

                    tgt_input = tgt_val[:, :-1]
                    tgt_output = tgt_val[:, 1:]
                    tgt_len = tgt_input.size(1)

                    tgt_pad_mask = create_padding_mask(tgt_input, pad_idx=en_vocab["<pad>"])
                    tgt_sub_mask = create_subsequent_mask(tgt_len).to(tgt_input.device)
                    tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(1)
                    tgt_mask = tgt_pad_mask & tgt_sub_mask

                    enc_output = model.encoder(src_val, src_mask)
                    logits = model.decoder(enc_output, tgt_input, tgt_mask=tgt_mask, memory_mask=memory_mask)

                    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

            # ==== 保存最佳模型 ====
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Best model saved at epoch {epoch + 1}, val_loss = {val_loss:.4f}")

    # 训练结束后保存最终模型（不一定是最优）
    torch.save(model.state_dict(), "last_model.pth")
    print("✅ Final model saved as last_model.pth")

# ------ 评估 ------
de_test_texts = [example['de'] for example in ds['test']]
en_test_texts = [example['en'] for example in ds['test']]
test_dataset = TranslationDataset(de_test_texts, en_test_texts, de_vocab, en_vocab)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

smoothie = SmoothingFunction().method4

def idx2word(vocab, idx):
    idx2word_map = {v: k for k, v in vocab.items()}
    return idx2word_map.get(idx, "<unk>")

# 贪婪解码器：逐步生成目标序列
def greedy_decode(model, src, src_mask, max_len, start_symbol, pad_idx):
    memory = model.encoder(src, src_mask)
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src)  # 初始 <bos>

    for _ in range(max_len - 1):
        tgt_mask = create_subsequent_mask(ys.size(1)).to(src.device)  # 自回归mask
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
        tgt_pad_mask = create_padding_mask(ys, pad_idx=pad_idx)
        full_tgt_mask = tgt_mask & tgt_pad_mask

        out = model.decoder(memory, ys, tgt_mask=full_tgt_mask, memory_mask=src_mask)
        next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

    return ys


# 模型评估（自动生成 + 计算 BLEU）
def evaluate(model, data_loader, device, max_len=50, verbose=False, load_model_path=None):
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
        print(f"✅ Model loaded from {load_model_path}")

    model.eval()
    references = []
    hypotheses = []

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (src_batch, tgt_batch) in enumerate(data_loader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            src_mask = create_padding_mask(src_batch, pad_idx=de_vocab["<pad>"])
            memory_mask = src_mask.clone()

            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            tgt_pad_mask = create_padding_mask(tgt_input, pad_idx=en_vocab["<pad>"])
            tgt_sub_mask = create_subsequent_mask(tgt_input.size(1)).to(device).unsqueeze(0).unsqueeze(1)
            tgt_mask = tgt_pad_mask & tgt_sub_mask

            output = model.decoder(
                model.encoder(src_batch, src_mask),
                tgt_input,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )

            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
            num_batches += 1

            # 生成翻译
            pred_seqs = greedy_decode(
                model, src_batch, src_mask,
                max_len=max_len,
                start_symbol=en_vocab["<bos>"],
                pad_idx=en_vocab["<pad>"]
            )

            for i, (pred, truth) in enumerate(zip(pred_seqs, tgt_batch)):
                pred_tokens = [
                    idx2word(en_vocab, idx.item())
                    for idx in pred
                    if idx.item() not in [en_vocab["<pad>"], en_vocab["<bos>"], en_vocab["<eos>"]]
                ]
                truth_tokens = [
                    idx2word(en_vocab, idx.item())
                    for idx in truth
                    if idx.item() not in [en_vocab["<pad>"], en_vocab["<bos>"], en_vocab["<eos>"]]
                ]

                hypotheses.append(pred_tokens)
                references.append([truth_tokens])

                if verbose and batch_idx == 0 and i < 3:
                    print(f"\n👉 Source {i + 1}:")
                    print(f"Predicted: {' '.join(pred_tokens)}")
                    print(f"Reference: {' '.join(truth_tokens)}")

    avg_loss = total_loss / num_batches
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100

    return avg_loss, bleu


if __name__ == '__main__':
    train_model(25, device, train_loader, val_loader)
    test_loss, test_bleu = evaluate(model, test_loader, device, verbose=True, load_model_path="last_model.pth")
    print(f"\n📊 Test Loss: {test_loss:.4f}")
    print(f"🔍 BLEU Score: {test_bleu:.2f}")