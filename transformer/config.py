from pathlib import Path

def get_config():
    return {
        "batch_size": 32,          # seq_len=128 时 3090 可以开到 32；seq_len 变大时再调小
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 128,            # OPUS-100 多为短句，512 会大量 padding，注意力还是 O(n²)
        "d_model": 512,
        "vocab_size": 16000,       # 5 万句对不必开 5 万词表；中英 BPE 16k 足够
        "max_train_samples": 50000,  # 只取前 5 万条平行句对做训练/验证
        "datasource": 'Helsinki-NLP/opus-100',
        "lang_src": "en",
        "lang_tgt": "zh",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def _weights_dir(config):
    # datasource 含 '/'（如 Helsinki-NLP/opus-100），不能直接当目录名
    ds = config["datasource"].replace("/", "_")
    return f"{ds}_{config['model_folder']}"

def get_weights_file_path(config, epoch: str):
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / _weights_dir(config) / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(_weights_dir(config)).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
