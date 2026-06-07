import torch
from torch import nn
import numpy as np
from data_utils import *
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from icecream import ic
import torch.nn.functional as F
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from models import GPT2RewardHead


class RewardModelTrainer():
    def __init__(self):
                
        model_name = 'gpt2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        REWARD_TOKEN_ID = self.tokenizer.eos_token_id
        self.tokenizer.add_special_tokens({
            "pad_token": "[PAD]"
        })

        # IMPORTANT: usually the exisitng works use pad from left but acc to the N+ paper implemenation we should pad from righ
        self.tokenizer.padding_side = "right"

        #! datasets
        ds = load_dataset("vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144")
        batch_size = 8
        train_dataset = PreferenceDataset(
            ds['train'],
            self.tokenizer
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            collate_fn = preference_collate_fn
        )


        eval_dataset = PreferenceDataset(
            ds['validation'],
            self.tokenizer
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size = batch_size,
            shuffle = True,
            collate_fn = preference_collate_fn
        )

        # ! we choose to use the finetuned sft model  
        sft_model_path = 'sft_epoch1_dnd/'
        self.model = GPT2RewardHead(sft_model_path=sft_model_path)

        self.EPOCHS       = 1
        LR           = 3e-6
        WARMUP_STEPS = 0 #! we are not using warmup
        self.grad_acc_steps = 8 # effective batch size 64
        self.optimizer = AdamW(self.model.parameters(), lr=LR, weight_decay=1e-5)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.EPOCHS,
        )

        device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model.llm.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(device)

    def get_reward(self, batch):

        query_responses = torch.cat(
            (
                batch["query_chosen_token"],
                batch["query_rejected_token"],
            ),
            dim=0,
        ).to(self.device)

        attention_mask = torch.cat(
            (
                batch["query_chosen_attention_mask"],
                batch["query_rejected_attention_mask"],
            ),
            dim=0,
        ).to(self.device)

        # this will return the mlp values for each token (aka reward for each), but we do not use the reward values of all tokens rather only 
        # the last one which is uses as <eos> or we can also use a special token
        # the <eos> token at the end of answer/summary (in this case) will be the eos token which will attend to all the previous tokens and can be 
        # used as final reward scaler.

        reward_logits = self.model(
            input_ids=query_responses,
            attention_mask=attention_mask,
        ) # shape: [2*bs, seq_len]

        # last position where attention_mask == 1, from this we take the eos index, we can also choose the just before padding directly
        # but the current padding is <query> <pad> <summary> so we choose to do this,
        eos_idx = (
            attention_mask.shape[1]
            - 1
            - attention_mask.flip(1).argmax(dim=1)
        )

        rewards = reward_logits[
            torch.arange(
                reward_logits.size(0),
                device=reward_logits.device,
            ),
            eos_idx,
        ]

        return rewards # this will return scaler for each batch, bs * 2, (bs1: positve summaries, bs2: negative summaries)
    
    def train(self):
        """
            This is the optimisation function for the reward model, 
            we use a mlp head about the transformer backbone (which was sft finetuned), the mlp layer will return the scaler values for each 
            token, then the get_reward() method will return the scaler values for each of the batch
            the accuracy for validation falls around ~60%, acc to the N+ implementation, 1B param models had ~65% accuracy, the gpt2 we use is 124M
            and we dont need a model with ~90sh accuracy, as the preferences are not well defined in general  
        """

        epoch_bar = tqdm(
            range(self.EPOCHS),
            desc="Training",
            position=0,
            leave=True
        )
        for epoch in range(self.EPOCHS):

            self.model.train()
            total_loss = 0.0
            self.optimizer.zero_grad()

            train_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.EPOCHS}",
                leave=False,
            )

            for step, batch in enumerate(train_bar):

                rewards = self.get_reward(self.model, batch) # this will return the 2*bs, reward one reward for each seq
                bs = batch["query_chosen_token"].shape[0]

                chosen_rewards = rewards[:bs]
                rejected_rewards = rewards[bs:]

                accuracy = (
                    chosen_rewards > rejected_rewards
                ).float().mean()

                loss = -F.logsigmoid(
                    chosen_rewards - rejected_rewards
                ).mean()

                total_loss += loss.item()

                # gradient accumulation
                loss = loss / self.grad_acc_steps
                loss.backward()

                if (step + 1) % self.grad_acc_steps == 0:

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0,
                    )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                train_bar.set_postfix(
                    loss=f"{loss.item() * self.grad_acc_steps:.4f}",
                    acc=f"{accuracy.item():.4f}",
                )

                # wandb.log(
                #     {
                #         "loss_train": loss.item() * grad_acc_steps,
                #         "accuracy_train": accuracy.item(),
                #     }
                # )

            # handle leftover gradients
            if len(self.train_loader) % self.grad_acc_steps != 0:

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.model.eval()

            eval_acc = []

            with torch.no_grad():

                eval_bar = tqdm(
                    self.eval_loader,
                    desc=f"Eval {epoch + 1}",
                    leave=False,
                )

                for step, batch in enumerate(eval_bar):

                    rewards = self.get_reward(self.model, batch)

                    bs = batch["query_chosen_token"].shape[0]

                    chosen_rewards = rewards[:bs]
                    rejected_rewards = rewards[bs:]

                    accuracy = (
                        chosen_rewards > rejected_rewards
                    ).float().mean()

                    eval_acc.append(
                        accuracy.item()
                    )

                    eval_bar.set_postfix(
                        acc=f"{accuracy.item():.4f}"
                    )

                mean_eval_acc = sum(eval_acc) / len(eval_acc)

                # wandb.log(
                #     {
                #         "eval_accuracy": mean_eval_acc,
                #         "epoch": epoch,
                #     }
                # )


            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {total_loss / len(self.train_loader):.4f} | "
                f"Eval Acc: {mean_eval_acc:.4f}"
            )
            torch.save(self.model.state_dict(), "reward_model.pt")


if __name__ == "__main__":
    inst = RewardModelTrainer()
    inst.train()