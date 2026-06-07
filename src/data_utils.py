import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from icecream import ic

MAX_QUERY_LEN = 512
MAX_RESP_LEN = 53


def format_query(subreddit, title, post):
    return (
        f"SUBREDDIT: r/{subreddit}\n\n"
        f"TITLE: {title}\n\n"
        f"POST: {post}\n\n"
        f"TL;DR:"
    )


def truncate_post_by_paragraph(subreddit, title, post, tokenizer):
    """
    Repeatedly remove last paragraph until query <= 512 tokens
    """

    while True:
        query = format_query(subreddit, title, post)

        tokenized = tokenizer(
            query,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        if len(tokenized) <= MAX_QUERY_LEN:
            return query

        last_newline = post.rfind("\n")

        if last_newline == -1:
            # fallback hard truncation
            return tokenizer.decode(
                tokenized[:MAX_QUERY_LEN],
                skip_special_tokens=True
            )

        post = post[:last_newline]

def truncate_query(query, tokenizer):
    """
    Repeatedly remove last paragraph until query <= 512 tokens
    """

    while True:
        # query = format_query(subreddit, title, post)

        tokenized = tokenizer(
            query,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        if len(tokenized) <= MAX_QUERY_LEN:
            return query

        last_newline = query.rfind("\n")

        if last_newline == -1:
            # fallback hard truncation
            return tokenizer.decode(
                tokenized[:MAX_QUERY_LEN],
                skip_special_tokens=True
            )

        query = query[:last_newline]

def preprocess_preference_examples(example, tokenizer):
    query = truncate_query(example["query"], tokenizer)

    query_enc = tokenizer(
        query,
        max_length=MAX_QUERY_LEN,
        padding="max_length",
        truncation=False,
        add_special_tokens=False,
    )

    query_tokens = query_enc["input_ids"]
    query_mask = query_enc["attention_mask"]

    choice = example["choice"]
    summaries = example["summaries"]

    # chosen
    chosen_res = " " + summaries[choice]["text"] + tokenizer.eos_token
    chosen_enc = tokenizer(
        chosen_res,
        max_length=MAX_RESP_LEN,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )

    chosen_tokens = chosen_enc["input_ids"]
    chosen_mask = chosen_enc["attention_mask"]

    # rejected
    rejected_res = " " + summaries[1 - choice]["text"] + tokenizer.eos_token
    rejected_enc = tokenizer(
        rejected_res,
        max_length=MAX_RESP_LEN,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )

    rejected_tokens = rejected_enc["input_ids"]
    rejected_mask = rejected_enc["attention_mask"]

    query_chosen_tokens = query_tokens + chosen_tokens
    query_rejected_tokens = query_tokens + rejected_tokens

    query_chosen_mask = query_mask + chosen_mask
    query_rejected_mask = query_mask + rejected_mask

    return {
        "query_token": torch.tensor(query_tokens),
        "chosen_token": torch.tensor(chosen_tokens),
        "rejected_token": torch.tensor(rejected_tokens),

        "query_chosen_token": torch.tensor(query_chosen_tokens),
        "query_rejected_token": torch.tensor(query_rejected_tokens),

        "query_chosen_attention_mask": torch.tensor(query_chosen_mask),
        "query_rejected_attention_mask": torch.tensor(query_rejected_mask),

        "choice": torch.tensor(choice),
    }


def preprocess_example(example, tokenizer):
    """  
        Dataset({
        features: ['id', 'subreddit', 'title', 'post', 'summary'],
        num_rows: 116722
        })
    """

    subreddit = example["subreddit"]
    title = example["title"]
    post = example["post"]
    summary = example["summary"]

    query = truncate_post_by_paragraph(
        subreddit,
        title,
        post,
        tokenizer,
    )

    query_tokens = tokenizer(
        query,
        max_length=MAX_QUERY_LEN,
        padding="max_length",
        truncation=False,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]

    response = " " + summary + tokenizer.eos_token

    response_tokens = tokenizer(
        response,
        max_length=MAX_RESP_LEN,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]

    query_response_tokens = query_tokens + response_tokens

    return {
        "query_token": query_tokens,
        "reference_response_token": response_tokens,
        "query_reference_response_token": query_response_tokens,
    }

class TLDRDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        processed = preprocess_example(
            example,
            self.tokenizer
        )

        return {
            "query_token": torch.tensor(
                processed["query_token"],
                dtype=torch.long
            ),
            "reference_response_token": torch.tensor(
                processed["reference_response_token"],
                dtype=torch.long
            ),
            "query_reference_response_token": torch.tensor(
                processed["query_reference_response_token"],
                dtype=torch.long
            ),
        }
    

class PreferenceDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        processed = preprocess_preference_examples(
            example,
            self.tokenizer
        )

        return {
            "query_token": torch.tensor(
                processed["query_token"],
                dtype=torch.long
            ),

            "chosen_token": torch.tensor(
                processed["chosen_token"],
                dtype=torch.long
            ),

            "rejected_token": torch.tensor(
                processed["rejected_token"],
                dtype=torch.long
            ),

            "query_chosen_token": torch.tensor(
                processed["query_chosen_token"],
                dtype=torch.long
            ),

            "query_rejected_token": torch.tensor(
                processed["query_rejected_token"],
                dtype=torch.long
            ),

            "query_chosen_attention_mask": torch.tensor(
                processed["query_chosen_attention_mask"],
                dtype=torch.long
            ),

            "query_rejected_attention_mask": torch.tensor(
                processed["query_rejected_attention_mask"],
                dtype=torch.long
            ),

            "choice": torch.tensor(
                processed["choice"],
                dtype=torch.long
            ),
        }

def collate_fn(batch):

    return {
        "query_token": torch.stack(
            [x["query_token"] for x in batch]
        ),
        "reference_response_token": torch.stack(
            [x["reference_response_token"] for x in batch]
        ),
        "query_reference_response_token": torch.stack(
            [x["query_reference_response_token"] for x in batch]
        ),
    }

def preference_collate_fn(batch):
    return {
        "query_chosen_token": torch.stack(
            [x["query_chosen_token"] for x in batch]
        ),
        "query_rejected_token": torch.stack(
            [x["query_rejected_token"] for x in batch]
        ),

        "query_chosen_attention_mask": torch.stack(
            [x["query_chosen_attention_mask"] for x in batch]
        ),
        "query_rejected_attention_mask": torch.stack(
            [x["query_rejected_attention_mask"] for x in batch]
        ),

        "choice": torch.stack(
            [x["choice"] for x in batch]
        ),
    }