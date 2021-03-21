# w2v

w2v in golang

```
    w2v <w2v_conf_path>

    {
        "lr":              0.01,
        "reg_rate":        0.01,
        "lr_down_rate":    0.95,
        "max_val":         1.0,
        "embedding_sz":    128,
        "min_words":       2,
        "window_sz":       2,
        "negatives":       2,
        "negative_e":      0.75,
        "embedding_reused": 1,
        "increase_training": 0,
        "iters":           50,
        "threads":         8,
        "model_type":      0,
        "train_path":      "../data/w2v_train",
        "embedding_path":  "../data//w2v_embedding.npy",
        "words_path":      "../data//w2v_words"
    }

```
