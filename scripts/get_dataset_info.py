import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import DATASETS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--info", required=True, choices=["base", "query", "groundtruth", "base_attrs", "query_attrs", "type", "basedir"])
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        sys.exit(1)

    ds = DATASETS[args.dataset]()
    
    if args.sample_size and hasattr(ds, 'set_subset'):
        ds.set_subset(args.sample_size)
    
    if args.info == "base":
        if hasattr(ds, "base_fn"):
            if os.path.isabs(ds.base_fn):
                print(ds.base_fn)
            else:
                print(os.path.join(ds.basedir, ds.base_fn))
        elif hasattr(ds, "get_dataset_fn"):
            print(ds.get_dataset_fn())
        elif hasattr(ds, "ds_fn"):
             print(os.path.join(ds.basedir, ds.ds_fn))
    elif args.info == "query":
        if hasattr(ds, "query_fn"):
            if os.path.isabs(ds.query_fn):
                print(ds.query_fn)
            else:
                print(os.path.join(ds.basedir, ds.query_fn))
        elif hasattr(ds, "qs_fn"):
             print(os.path.join(ds.basedir, ds.qs_fn))
    elif args.info == "groundtruth":
        if hasattr(ds, "gt_fn"):
            if os.path.isabs(ds.gt_fn):
                path = ds.gt_fn
            else:
                path = os.path.join(ds.basedir, ds.gt_fn)
            
            # Only return the path if it exists. 
            # If it doesn't exist (e.g. subset without GT), return empty string
            # so the C++ backend knows to generate it.
            if os.path.exists(path):
                print(path)
    elif args.info == "basedir":
        print(ds.basedir)
    elif args.info == "base_attrs":
        if hasattr(ds, "ds_metadata_fn"):
             print(os.path.join(ds.basedir, ds.ds_metadata_fn))
        elif hasattr(ds, "base_attrs_path"):
             print(ds.base_attrs_path)
    elif args.info == "query_attrs":
        if hasattr(ds, "qs_metadata_fn"):
            print(os.path.join(ds.basedir, ds.qs_metadata_fn))
        elif hasattr(ds, "query_attrs_path"):
            print(ds.query_attrs_path)
    elif args.info == "type":
        print(ds.data_type())

if __name__ == "__main__":
    main()
