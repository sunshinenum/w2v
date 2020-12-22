import os
import faiss
import numpy as np

words_path = "../data/w2v_words"
embeddings_path = "../data/w2v_embedding.npy"
words_sims_checks = 1000
sims_check = 50
result_path = "../data/words_sims_check"

words = os.popen("cut -f -1 {}".format(words_path)).read().strip("\n").split("\n")
embeddings = np.load(embeddings_path)
embeddings = np.asarray(embeddings, dtype=np.float32)

if len(words) != len(embeddings):
    print("len(words) != len(embeddings) !")
    exit(-1)

index = faiss.IndexFlatIP(len(embeddings[0]))   # build the index
index.add(embeddings)

D, I = index.search(embeddings[:words_sims_checks], sims_check)
with open(result_path, "w") as op:
    for i, scores in enumerate(D):
        indexes = I[i]
        words_cu = words[i]
        line = [words_cu + "\t"]
        for j in range(len(scores)):
            line.append("{} {:.4f}".format(words[indexes[j]], scores[j]))
        op.write("{}\n".format(" ".join(line)))
