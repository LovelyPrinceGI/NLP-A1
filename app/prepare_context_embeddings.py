import numpy as np
import torch
import pickle
from models import GloVeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) load vocab
with open("word2index.pkl", "rb") as f:
    word2index = pickle.load(f)

# 2) load corpus sentences
with open("corpus_sentences.pkl", "rb") as f:
    corpus_sentences = pickle.load(f)   # นี่แหละคือ corpus เดิมของคุณ

# 3) load GloVe model
model_glove = GloVeModel(vocab_size=len(word2index), emb_size=100)
model_glove.load_state_dict(torch.load("glove_model.pt", map_location=device))
model_glove.to(device)
model_glove.eval()

def get_word_vec(word):
    if word not in word2index:
        return None
    idx = torch.tensor([word2index[word]]).to(device)
    with torch.no_grad():
        v = model_glove.v_embed(idx)
        if hasattr(model_glove, "u_embed"):
            u = model_glove.u_embed(idx)
            v = (v + u) / 2
    return v.cpu().numpy().squeeze()

def sentence_embedding(tokens):
    vecs = []
    for w in tokens:
        v = get_word_vec(w)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

context_vectors = []
context_texts = []

for sent in corpus_sentences:
    emb = sentence_embedding(sent)
    if emb is not None:
        context_vectors.append(emb)
        context_texts.append(" ".join(sent))

context_vectors = np.stack(context_vectors, axis=0)

np.save("context_vectors.npy", context_vectors)
with open("context_texts.txt", "w", encoding="utf-8") as f:
    for s in context_texts:
        f.write(s + "\n")

print("Saved context_vectors.npy and context_texts.txt")
