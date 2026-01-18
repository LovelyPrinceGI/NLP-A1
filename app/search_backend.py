# search_backend.py (รองรับหลายโมเดล)
import torch, pickle, numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("word2index.pkl", "rb") as f:
    word2index = pickle.load(f)

from models import GloVeModel, SkipgramNegSampling

models = {}

# GloVe
models["glove"] = GloVeModel(len(word2index), 100).to(device)
models["glove"].load_state_dict(torch.load("glove_model.pt", map_location=device))
models["glove"].eval()

# Skip-gram (NS)
models["skipgram_ns"] = SkipgramNegSampling(len(word2index), 100).to(device)
models["skipgram_ns"].load_state_dict(torch.load("skipgram_ns.pt", map_location=device))
models["skipgram_ns"].eval()

context_vectors = np.load("context_vectors.npy")
with open("context_texts.txt", encoding="utf-8") as f:
    context_texts = [line.strip() for line in f]

def get_word_vec(word, model_name="glove"):
    if word not in word2index:
        return None
    idx = torch.tensor([word2index[word]]).to(device)
    model = models[model_name]
    with torch.no_grad():
        v = model.v_embed(idx)
        if hasattr(model, "u_embed"):
            u = model.u_embed(idx)
            v = (v + u) / 2
    return v.cpu().numpy().squeeze()

def query_embedding(query, model_name="glove"):
    tokens = query.lower().split()
    vecs = []
    for w in tokens:
        v = get_word_vec(w, model_name)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def search_topk(query, k=10, model_name="glove"):
    q_vec = query_embedding(query, model_name)
    if q_vec is None:
        return []
    scores = context_vectors @ q_vec
    topk_idx = np.argsort(-scores)[:k]
    return [{"text": context_texts[i], "score": float(scores[i])} for i in topk_idx]
