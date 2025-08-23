import argparse
from typing import cast

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.utils.dataset_transformation import load_train_set


def parse_args():
    parser = argparse.ArgumentParser(description="Compute hard negatives for ColPali training")

    parser.add_argument("--compute-embeddings", action="store_true", help="Compute document embeddings")
    parser.add_argument("--compute-hardnegs", action="store_true", help="Compute hard negatives from embeddings")
    parser.add_argument("--compute-topk-percpos", action="store_true", help="Create TopK-PercPos filtered dataset")
    parser.add_argument(
        "--perc-margin", type=float, default=0.95, help="Percentage margin for TopK-PercPos filtering (default: 0.95)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Convert args to the original variable names
    COMPUTE_EMBEDDINGS = args.compute_embeddings
    COMPUTE_HARDNEGS = args.compute_hardnegs
    COMPUTE_TOPK_PERCPOS = args.compute_topk_percpos
    PERC_MARGIN = args.perc_margin

    train_set = load_train_set()

    if COMPUTE_HARDNEGS or COMPUTE_EMBEDDINGS:
        print("Loading base model")
        model = ColQwen2_5.from_pretrained(
            "vidore/colqwen2.5-base",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        ).eval()

        print("Loading processor")
        processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-base")

    if COMPUTE_EMBEDDINGS:
        print("Loading images")

        document_set = train_set.data  # ["train"]
        print("Filtering dataset")
        print(document_set)
        initial_list = document_set["image_filename"]
        _, unique_indices = np.unique(initial_list, return_index=True, axis=0)
        filtered_dataset = document_set.select(unique_indices.tolist())
        filtered_dataset = filtered_dataset.map(
            lambda example: {"image": example["image"], "image_filename": example["image_filename"]}, num_proc=16
        )
        # keep only column image and image_filename and source if it exists
        cols_to_remove = [col for col in filtered_dataset.column_names if col not in ["image", "image_filename"]]
        filtered_dataset = filtered_dataset.remove_columns(cols_to_remove)
        # save it
        print("Saving filtered dataset")
        print(filtered_dataset)
        filtered_dataset.save_to_disk("data_dir/filtered_dataset", max_shard_size="200MB")

        print("Processing images")
        # run inference - docs
        dataloader = DataLoader(
            filtered_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda x: processor.process_images([a["image"] for a in x]),
        )
        print("Computing embeddings")

        ds = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

        # Find the maximum sequence length
        max_len = max(emb.size(0) for emb in ds)

        # Pad all embeddings to the same length
        padded_ds = []
        for emb in ds:
            if emb.size(0) < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - emb.size(0), emb.size(1), dtype=emb.dtype)
                emb = torch.cat([emb, padding], dim=0)
            padded_ds.append(emb)

        ds = torch.stack(padded_ds)

        # save embeddings
        torch.save(ds, "data_dir/filtered_dataset_embeddings.pt")

    if not COMPUTE_EMBEDDINGS:
        ds = torch.load("data_dir/filtered_dataset_embeddings.pt")

    if COMPUTE_HARDNEGS:
        # compute hard negatives
        ds = cast(torch.Tensor, ds).to("cuda")

        # iterate on the train set
        mined_hardnegs = []
        mined_scores = []  # Store scores for TopK-PercPos

        for i in tqdm(range(0, len(train_set.data), 8)):
            samples = train_set.data[i : i + 8]
            batch_query = processor.process_queries(samples["query"])
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embeddings_query = model(**batch_query)

            # compute scores using ColBERT late interaction (same as loss function)
            # embeddings_query: (batch_size, num_query_tokens, dim)
            # ds: (num_docs, num_doc_tokens, dim)
            batch_size = embeddings_query.size(0)

            # Expand ds to match batch dimension
            ds_expanded = ds.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch_size, num_docs, seq_len, dim)

            # ColBERT scoring: for each query token, find max similarity with any doc token, then sum across query tokens
            scores = torch.einsum("bnd,bmsd->bmns", embeddings_query, ds_expanded).max(dim=3)[0].sum(dim=2)

            # get top 100 indexes and scores
            top100_scores, top100_indices = scores.topk(100, dim=1)
            # convert to lists
            top100_indices = top100_indices.tolist()
            top100_scores = top100_scores.tolist()
            # append to mined data
            mined_hardnegs.extend(top100_indices)
            mined_scores.extend(top100_scores)

        # save mined hardnegs as txt
        with open("data_dir/mined_hardnegs_filtered.txt", "w") as f:
            for item in mined_hardnegs:
                f.write("%s\n" % item)

        # save mined scores as txt
        with open("data_dir/mined_scores_filtered.txt", "w") as f:
            for item in mined_scores:
                f.write("%s\n" % item)

    # Load data for dataset creation
    with open("data_dir/mined_hardnegs_filtered.txt") as f:
        mined_hardnegs = f.readlines()

    with open("data_dir/mined_scores_filtered.txt") as f:
        mined_scores = f.readlines()

    filtered_dataset = datasets.load_from_disk("data_dir/filtered_dataset")
    filenames = list(filtered_dataset["image_filename"])

    # Original naive top-k mapper
    def mapper_fn(example, idx):
        tmp = {
            "negative_passages": [int(x) for x in mined_hardnegs[idx][1:-2].strip().split(",")],
            "query": example["query"],
            "positive_passages": [filenames.index(example["image_filename"])],
        }

        tmp["gold_in_top_100"] = tmp["positive_passages"][0] in tmp["negative_passages"]
        # remove gold index from negs if it is there
        if tmp["gold_in_top_100"]:
            tmp["negative_passages"].remove(tmp["positive_passages"][0])
        return tmp

    # TopK-PercPos mapper
    def topk_percpos_mapper(example, idx):
        """TopK-PercPos negatives filter implementation"""
        # Parse the mined negatives and scores for this example
        neg_indices = [int(x) for x in mined_hardnegs[idx][1:-2].strip().split(",")]
        neg_scores = [float(x) for x in mined_scores[idx][1:-2].strip().split(",")]

        # Get positive passage index
        pos_idx = filenames.index(example["image_filename"])

        # Find positive score and remove it from negatives
        if pos_idx in neg_indices:
            pos_score_idx = neg_indices.index(pos_idx)
            pos_score = neg_scores[pos_score_idx]
            neg_indices.pop(pos_score_idx)
            neg_scores.pop(pos_score_idx)
        else:
            pos_score = max(neg_scores) if neg_scores else 1.0

        # Apply TopK-PercPos filter: keep negatives with score < pos_score * margin
        threshold = pos_score * PERC_MARGIN
        original_count = len(neg_indices)
        filtered_negatives = [neg_idx for neg_idx, neg_score in zip(neg_indices, neg_scores) if neg_score < threshold]

        # Fallback: ensure at least one negative
        if not filtered_negatives and neg_indices:
            min_score_idx = neg_scores.index(min(neg_scores))
            filtered_negatives = [neg_indices[min_score_idx]]

        # Debug: print stats for first few examples
        if idx < 5:
            print(f"Example {idx}: {original_count} → {len(filtered_negatives)} negatives (threshold: {threshold:.3f})")

        return {
            "negative_passages": filtered_negatives,
            "query": example["query"],
            "positive_passages": [pos_idx],
            "gold_in_top_100": pos_idx in [int(x) for x in mined_hardnegs[idx][1:-2].strip().split(",")],
        }

    # Create and save naive top-k dataset
    final_dataset = train_set.data.map(mapper_fn, with_indices=True, num_proc=16)
    final_dataset = final_dataset.remove_columns("image")
    final_dataset.save_to_disk("data_dir/naive_topk_dataset")

    # Create and save TopK-PercPos dataset
    if COMPUTE_TOPK_PERCPOS:
        print(f"Creating TopK-PercPos dataset with margin {PERC_MARGIN}")
        topk_percpos_dataset = train_set.data.map(topk_percpos_mapper, with_indices=True, num_proc=16)
        topk_percpos_dataset = topk_percpos_dataset.remove_columns("image")
        topk_percpos_dataset.save_to_disk(f"data_dir/topk_percpos_dataset_margin_{PERC_MARGIN}")
        print(f"Saved TopK-PercPos dataset with margin {PERC_MARGIN}")


if __name__ == "__main__":
    main()
