import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset
from data_utils import *
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb


class SFTTrainer():
    def __init__(self):
        model_name = 'gpt2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-1b"
        )
        self.tokenizer.add_special_tokens({
            "pad_token": "[PAD]"
        })
        # IMPORTANT: usually the exisitng works use pad from left but acc to the N+ paper implemenation we should pad from righ
        self.tokenizer.padding_side = "right"


        # ! dataset
        dataset = load_dataset(
            "vwxyzjn/summarize_from_feedback_tldr_3_filtered",
            # split="train"
        )

        train_dataset = TLDRDataset(
            dataset['train'],
            self.tokenizer
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_fn,
        )

        test_dataset = TLDRDataset(
            dataset['test'],
            self.tokenizer
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # training params
        self.EPOCHS       = 1
        LR           = 3e-6
        # WARMUP_STEPS = 200 warmup was not used in the N+ implementation
        self.gradient_accumulation_steps = 4
        optimizer = AdamW(self.model.parameters(), lr=LR, weight_decay=1e-5)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.EPOCHS,
        )

        self.device = 'cuda'
        self.model.resize_token_embeddings(len(self.tokenizer))
    
        wandb.init(
            project="gpt2_finetuning",
            name=f"gpt2",
            mode="online",
            # config={
            #     "stage": stage,
            #     "learning_rate": lr,
            #     "epochs": self.NUM_EPOCHS,
            #     "grad_accum": self.GRAD_ACCUM_STEPS,
            #     "dataset": "localized_narratives",
            # }
        )

    def sft_loss(self,batch):

        ids = batch["query_reference_response_token"] # this is the query + response token, the seuqnece is right padded

        input_ids = ids.to(self.device)
        labels    = input_ids.clone()

        query_len = batch["query_token"].shape[1]          # 512 — mask the entire prompt
        labels[:, :query_len] = -100                       # mask query
        labels[input_ids == self.tokenizer.pad_token_id] = -100 # mask right-side padding

        return self.model(input_ids=input_ids, labels=labels).loss
    
    def train(self):
        self.model = self.model.to(self.device)

        for epoch in range(self.EPOCHS):

            # train
            self.model.train()
            total_loss = 0

            self.optimizer.zero_grad()

            for step, batch in tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch+1} train"
            ):

                loss = self.sft_loss(self.model, batch)

                # normalize loss for accumulation
                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (
                    (step + 1) % self.gradient_accumulation_steps == 0
                    or (step + 1) == len(self.train_loader)
                ):

                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps

                wandb.log(
                    {
                        "loss": loss.item() * self.gradient_accumulation_steps,
                        "step": step
                    }
                )

            # eval
            self.model.eval()
            eval_loss = 0

            # tr
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc=f"Epoch {epoch+1} eval"):
                    # batch = batch.to(device)
                    eval_loss += self.sft_loss(self.model, batch).item()

            print(
                f"Epoch {epoch+1} | train_loss: {total_loss/len(self.train_loader):.4f} "
                f"| eval_loss: {eval_loss/len(self.test_loader):.4f}"
            )

            self.model.save_pretrained(f"sft_epoch{epoch+1}")
            self.tokenizer.save_pretrained(f"sft_epoch{epoch+1}")

if __name__ == "__main__":
    inst = SFTTrainer()
    inst.train()