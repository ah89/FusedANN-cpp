import argparse
import sys
import os

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DATASETS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    ds = DATASETS[args.dataset]()
    print(f"Preparing {args.dataset}...")
    
    if args.sample_size:
        if hasattr(ds, 'set_subset'):
            print(f"Setting subset size to {args.sample_size}")
            ds.set_subset(args.sample_size)
            if hasattr(ds, 'prepare_subset'):
                ds.prepare_subset()
            else:
                 print("Dataset does not support prepare_subset, running standard prepare")
                 ds.prepare()
        else:
             print("Dataset does not support sampling, running standard prepare")
             ds.prepare()
    else:
        ds.prepare()
    
    print("Done.")

if __name__ == "__main__":
    main()
