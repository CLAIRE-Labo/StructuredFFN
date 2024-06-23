from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from argparse import ArgumentParser
import tiktoken
import os
from itertools import chain
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizer

long_path = "/claire-rcp-scratch/shared/xwei/dataset/tiiuae___falcon-refinedweb/default-4033b99bd924aaad/0.0.0/0111277fb19b16f696664cde7f0cb90f833dec72db2cc73cfdf87e697f78fe02"
cache_dir = "/claire-rcp-scratch/shared/xwei/dataset"


def tokenize(tokenizer, num_proc, dataset):
    if tokenizer == "gpt2":
        enc = tiktoken.get_encoding("gpt2")

        def tokenize_process(example):
            ids = enc.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                enc.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {"ids": ids}
            return out

    elif tokenizer == "llama":
        enc = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
        eos_tokens = enc(
            "</s>", truncation=False, padding=False, add_special_tokens=False
        )["input_ids"]

        def tokenize_process(example):
            ids = enc(
                example["text"],
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            ids = ids + eos_tokens
            out = {"ids": ids}
            return out

    else:
        raise NotImplementedError

    tokenized = dataset.map(
        tokenize_process,
        remove_columns=["text", "url", "timestamp", "dump", "segment", "image_urls"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    print(tokenized)
    return tokenized


def group_context(block_size, num_proc, dataset):

    def group_process(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = dataset.map(
        group_process,
        batched=True,
        num_proc=num_proc,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    print(lm_datasets)
    return lm_datasets


def save_to_npmemmap(split, dset, tokenizer, block_size):
    arr_len = dset.num_rows
    print(split, arr_len)
    filename = os.path.join(
        os.path.join(cache_dir, "refinedweb"), f"{tokenizer}-{split}-tmp.bin"
    )
    dtype = np.uint16  # (can do since enc.max_token_value == 32000 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len, block_size))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        # Batch together samples for faster write
        batch = dset.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        # Write into mmap
        arr_batch = np.stack(batch["ids"])
        arr[idx : idx + arr_batch.shape[0], :] = arr_batch
        idx += arr_batch.shape[0]
    arr.flush()


def parse_args():
    parser = ArgumentParser(
        description="Convert dataset into MDS format, optionally concatenating and tokenizing"
    )
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument(
        "--block_size",
        type=int,
        help="Convert text to tokens and concatenate up to this many tokens",
    )

    parser.add_argument("--num_proc", type=int, required=True, default=None)
    return parser.parse_args()


def main(args):
    print(args.num_proc)
    new_dataset = []
    for i in range(6):
        for j in range(10):
            if i == 5 and j > 3:
                continue
            refinedweb_chunk = load_dataset(
                path=long_path,
                split="train",
                data_files=f"falcon-refinedweb-train-0{i}{j}*-of-05379.arrow",
                num_proc=args.num_proc,
            ).shuffle(seed=i * 10 + j)
            print(refinedweb_chunk)
            total_rows = refinedweb_chunk.num_rows
            selected_rows = int(0.1 * total_rows)
            cur_chunk = refinedweb_chunk.select(range(selected_rows)).rename_column(
                "content", "text"
            )
            del refinedweb_chunk
            print("begin to tokenize!")
            # tokenization
            cur_chunk = tokenize(args.tokenizer, args.num_proc, cur_chunk)
            cur_chunk = group_context(args.block_size, args.num_proc, cur_chunk)
            new_dataset.append(cur_chunk)

    new_dataset = concatenate_datasets(new_dataset)
    new_dataset = new_dataset.train_test_split(test_size=0.01, seed=1005, shuffle=True)

    save_to_npmemmap("train", new_dataset["train"], args.tokenizer, args.block_size)
    save_to_npmemmap("val", new_dataset["test"], args.tokenizer, args.block_size)


if __name__ == "__main__":
    main(parse_args())
