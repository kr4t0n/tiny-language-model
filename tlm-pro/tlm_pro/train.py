import torch
import wandb
import argparse
import os.path as osp

from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader

from modeling_tlm import TLM
from data_utils import prepare_data

args_parser = argparse.ArgumentParser("tlm-pro")
args_parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
args_parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b")
args_parser.add_argument("--vocab_size", type=int, default=32000)
args_parser.add_argument("--max_length", type=int, default=512)
args_parser.add_argument("--n_layers", type=int, default=12)
args_parser.add_argument("--d_model", type=int, default=1024)
args_parser.add_argument("--n_heads", type=int, default=8)
args_parser.add_argument("--dropout", type=float, default=0.2)
args_parser.add_argument("--learning_rate", type=float, default=1e-4)
args_parser.add_argument("--min_learning_rate", type=float, default=1e-5)
args_parser.add_argument("--weight_decay", type=float, default=1e-3)
args_parser.add_argument("--num_epochs", type=int, default=10)
args_parser.add_argument("--batch_size", type=int, default=32)
args_parser.add_argument("--log_interval", type=int, default=10)
args_parser.add_argument("--eval_interval", type=int, default=500)
args_parser.add_argument("--eval_prompt", type=str, default="Once upon a time,")
args_parser.add_argument("--ckpt_interval", type=int, default=1000)
args_parser.add_argument("--output_dir", type=str, default="./")
args_parser.add_argument("--ckpt_dir", type=str, default=None)
args_parser.add_argument("--ckpt_step", type=int, default=-1)
args = args_parser.parse_args()


def main():
    accelerator = Accelerator()

    train_dataset, valid_dataset, tokenizer, data_collator = prepare_data(
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        num_workers=4,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        num_workers=4,
    )

    model = TLM(
        vocab_size=args.vocab_size,
        n_layers=args.n_layers,
        context_length=args.max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.min_learning_rate,
    )

    train_dataloader, valid_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, valid_dataloader, model, optimizer, scheduler
    )

    # load checkpoint
    if args.ckpt_dir is not None:
        accelerator.load_state(args.ckpt_dir)

    if accelerator.is_main_process:
        pbar = tqdm()
        wandb.init(project="tlm-pro")
        wandb.watch(model, accelerator.unwrap_model(model).loss_fn, log="all")

    step = args.ckpt_step
    for epoch in range(args.num_epochs):
        train_dataset.set_epoch(epoch=epoch)

        for inputs, targets in train_dataloader:
            step += 1

            model.train()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = accelerator.unwrap_model(model).loss_fn(outputs, targets)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            if step % args.eval_interval == 0:
                model.eval()

                valid_loss, valid_n = 0.0, 0
                for valid_inputs, valid_targets in valid_dataloader:
                    valid_outputs = model(valid_inputs)
                    loss = accelerator.unwrap_model(model).loss_fn(valid_outputs, valid_targets)

                    valid_n += valid_inputs.shape[0]
                    valid_loss += loss.item() * valid_n

                valid_loss = valid_loss / valid_n

            if accelerator.is_main_process:
                if step % args.log_interval == 0:
                    pbar.set_description(f"Epoch: {epoch}, step: {step}, loss: {loss.item()}")
                    pbar.update(step)
                    wandb.log({"train_loss": loss.item()}, step=step)

                if step % args.eval_interval == 0:
                    wandb.log({"valid_loss": valid_loss}, step=step)

                    eval_generation = accelerator.unwrap_model(model).generate(
                        args.eval_prompt,
                        tokenizer,
                        max_length=args.max_length,
                        device=accelerator.device,
                    )
                    wandb.log(
                        {
                            "generations": wandb.Table(
                                columns=["prompt", "generation"],
                                data=[[args.eval_prompt, eval_generation]],
                            )
                        },
                        step=step,
                    )

                if step % args.ckpt_interval == 0:
                    accelerator.save_state(output_dir=osp.join(args.output_dir, f"ckpt-{step}"))

        if accelerator.is_main_process:
            accelerator.save_state(output_dir=osp.join(args.output_dir, f"epoch-{epoch}"))

    if accelerator.is_main_process:
        accelerator.save_model(accelerator.unwrap_model(model), osp.join(args.output_dir, "tlm-pro"))

    if accelerator.is_main_process:
        pbar.close()
        wandb.unwatch(model)
        wandb.finish()


if __name__ == "__main__":
    main()
