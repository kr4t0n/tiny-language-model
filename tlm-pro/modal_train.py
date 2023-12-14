import modal

from dataclasses import dataclass

LOCAL_DIR = "./tlm_pro"
REMOTE_DIR = "/root/tlm_pro"
REMOTE_VOL_DIR = "/root/tlm_pro_vol"


def download_dataset():
    from datasets import load_dataset

    load_dataset("roneneldan/TinyStories")


stub = modal.Stub("tlm-pro")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "sentencepiece",
        "transformers",
        "datasets",
        "accelerate",
        "wandb",
    )
    .run_function(download_dataset)
)
tlm_pro_vol = modal.NetworkFileSystem.new().persist("tlm-pro-vol")


@dataclass
class TrainArguments:
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "meta-llama/Llama-2-7b"

    vocab_size: int = 32000
    max_length: int = 512
    n_layers: int = 12
    d_model: int = 1024
    n_heads: int = 8
    dropout: float = 0.2

    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-3

    num_epochs: int = 4
    batch_size: int = 16

    log_interval: int = 10
    ckpt_interval: int = 800

    output_dir: str = REMOTE_VOL_DIR


@stub.function(
    image=image,
    gpu="A10G",
    mounts=[
        modal.Mount.from_local_dir(LOCAL_DIR, remote_path=REMOTE_DIR),
    ],
    network_file_systems={
        REMOTE_VOL_DIR: tlm_pro_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=86400,
)
def modal_train():
    import subprocess

    from accelerate.utils import write_basic_config

    # set up huggingface accelerate basic config
    write_basic_config(mixed_precision="fp16")

    # set up train args
    args = TrainArguments()

    # train
    subprocess.run(
        [
            "accelerate",
            "launch",
            f"{REMOTE_DIR}/train.py",
            "--dataset_name",
            f"{args.dataset_name}",
            "--tokenizer_name",
            f"{args.tokenizer_name}",
            "--vocab_size",
            f"{args.vocab_size}",
            "--max_length",
            f"{args.max_length}",
            "--n_layers",
            f"{args.n_layers}",
            "--d_model",
            f"{args.d_model}",
            "--n_heads",
            f"{args.n_heads}",
            "--dropout",
            f"{args.dropout}",
            "--learning_rate",
            f"{args.learning_rate}",
            "--min_learning_rate",
            f"{args.min_learning_rate}",
            "--weight_decay",
            f"{args.weight_decay}",
            "--num_epochs",
            f"{args.num_epochs}",
            "--batch_size",
            f"{args.batch_size}",
            "--log_interval",
            f"{args.log_interval}",
            "--ckpt_interval",
            f"{args.ckpt_interval}",
            "--output_dir",
            f"{args.output_dir}",
        ],
        check=True,
    )


@stub.local_entrypoint()
def main():
    modal_train.remote()
