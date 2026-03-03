import argparse
import json
from tqdm import tqdm

from synctxasr.data import dataset_functions


def count_words(text: str) -> int:
    return len(text.strip().split()) if text and text.strip() else 0


def main(args):
    assert args.dataset in dataset_functions, (
        f"Dataset {args.dataset} not found, available datasets: {list(dataset_functions.keys())}"
    )

    dataset_fn = dataset_functions[args.dataset]
    dataset = dataset_fn(args.split)

    if sum(args.indexes) == -1:
        indexes = list(range(len(dataset)))
    else:
        indexes = args.indexes

    total_utterances = 0
    total_words = 0

    for i in tqdm(indexes, desc="Processing recordings", unit="rec"):
        recording = dataset[i]
        utterances = recording["process_fn"](recording)

        for utt in utterances:
            words = count_words(utt.get("text", ""))
            if args.exclude_empty and words == 0:
                continue
            total_words += words
            total_utterances += 1

    average_words = (total_words / total_utterances) if total_utterances > 0 else 0.0

    output_lines = [
        f"Dataset: {args.dataset}",
        f"Split: {args.split}",
        f"Recordings processed: {len(indexes)}",
        f"Total utterances counted: {total_utterances}",
        f"Total words: {total_words}",
        f"Average words per utterance: {average_words:.4f}",
    ]
    output_text = "\n".join(output_lines)
    print(output_text)

    if args.output_file is not None:
        if args.output_format == "avg":
            file_text = f"{average_words:.4f}\n"
        elif args.output_format == "json":
            file_text = json.dumps(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "recordings_processed": len(indexes),
                    "total_utterances_counted": total_utterances,
                    "total_words": total_words,
                    "average_words_per_utterance": round(average_words, 4),
                },
                indent=2,
            ) + "\n"
        else:
            file_text = output_text + "\n"

        with open(args.output_file, "w") as f:
            f.write(file_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average number of words per utterance for a dataset split."
    )
    parser.add_argument("--dataset", type=str, default="tedlium3")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--indexes",
        type=int,
        nargs="+",
        default=[-1],
        help="Indexes of recordings to process; -1 means all.",
    )
    parser.add_argument(
        "--exclude_empty",
        action="store_true",
        help="Exclude empty utterances (0 words) from the average.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save the summary output.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["avg", "json", "text"],
        default="avg",
        help="Format used when writing --output_file.",
    )

    args = parser.parse_args()
    main(args)
