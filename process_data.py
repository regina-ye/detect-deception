import pandas as pd
from typing import Dict, List, Optional
import os
from pathlib import Path
import pickle
from datasets import load_dataset


class DataLoader:
    """Loads and processes DIFrauD dataset from HuggingFace"""

    def __init__(self, save_dir: str = "processed_data"):
        self.save_dir = save_dir
        self.domains = [
            "fake_news",
            "job_scams",
            "phishing",
            "political_statements",
            "product_reviews",
            "sms",
            "twitter_rumours",
        ]

    def load_domain_data(
        self, domain: str, sample_size: Optional[int] = None
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load train, test, and validation data for a domain
        from HuggingFace
        """
        print(f"Loading data for domain: {domain}")

        try:
            # load dataset from HuggingFace
            dataset = load_dataset("redasers/difraud", domain)
            if not dataset:
                print(f"no data found for domain: {domain}")
                return None

            data_dict = {}

            # convert to DataFrames and sample if necessary
            for split in dataset.keys():
                df = pd.DataFrame(dataset[split])

                # make sure required columns exist
                if "text" not in df.columns or "label" not in df.columns:
                    print(f"missing required columns in {domain} {split}")
                    continue

                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)

                data_dict[split] = df
                print(
                    f"  {split}: loaded {len(df)} samples "
                    f"({df['label'].sum()} deceptive, "
                    f"{len(df) - df['label'].sum()} non-deceptive)"
                )

            return data_dict if data_dict else None

        except Exception as e:
            print(f"error loading domain {domain}: {str(e)}")
            return None

    def validate_data(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> bool:
        """Ensure loaded data is not empty and has correct structure"""
        if not data:
            return False

        for domain, splits in data.items():
            if not splits:
                continue
            for split, df in splits.items():
                if (
                    df is None
                    or df.empty
                    or "text" not in df.columns
                    or "label" not in df.columns
                ):
                    return False
        return True

    def load_all_domains(
        self, sample_size: Optional[int] = None, force_reload: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load data for all domains"""
        # try loading from disk first if not forcing reload
        if not force_reload and os.path.exists(
            os.path.join(self.save_dir, "processed_data.pkl")
        ):
            print("found existing processed data, will try to load...")
            data = self.load_saved_data()
            if data and self.validate_data(data):
                print("successfully loaded data from disk")
                return data
            else:
                print("saved data was invalid, reloading...")

        print("loading data from HuggingFace...")
        all_data = {}
        for domain in self.domains:
            domain_data = self.load_domain_data(domain, sample_size)
            if domain_data:
                all_data[domain] = domain_data

        if not self.validate_data(all_data):
            raise ValueError("failed to load data from HuggingFace")

        return all_data

    def save_processed_data(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        """Save processed data and metadata to disk"""
        if not self.validate_data(data):
            print("you are trying to save invalid data")
            return

        #create directory to save in
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # save main data
        save_path = os.path.join(self.save_dir, "processed_data.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        # save metadata
        metadata_path = os.path.join(self.save_dir, "metadata.txt")
        with open(metadata_path, "w") as f:
            f.write("Dataset info:\n")
            for domain, splits in data.items():
                f.write(f"\n=== {domain} ===\n")
                for split, df in splits.items():
                    if df is not None and not df.empty:
                        deceptive = df["label"].sum()
                        total = len(df)
                        f.write(f"\n{split}:\n")
                        f.write(f"  total samples: {total}\n")
                        f.write(
                            f"  deceptive: {deceptive} "
                            f"({deceptive / total * 100:.1f}%)\n"
                        )
                        f.write(
                            f"  non-deceptive: {total-deceptive} "
                            f"({(total-deceptive)/total*100:.1f}%)\n"
                        )

    def load_saved_data(self) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
        """Load processed data from disk"""
        save_path = os.path.join(self.save_dir, "processed_data.pkl")
        try:
            with open(save_path, "rb") as f:
                data = pickle.load(f)
            if self.validate_data(data):
                return data
            else:
                print("loaded data is invalid")
                return None
        except Exception as e:
            print(f"error loading data: {str(e)}")
            return None

    def print_statistics(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        """Print dataset statistics"""
        if not data:
            print("no data to show statistics for")
            return

        for domain, splits in data.items():
            if not splits:
                continue

            print(f"\n=== {domain} ===")
            for split_name, df in splits.items():
                if df is not None and not df.empty:
                    deceptive = df["label"].sum()
                    total = len(df)
                    print(f"\n{split_name}:")
                    print(f"total samples: {total}")
                    print(
                        f"deceptive samples: {deceptive} "
                        f"({deceptive / total * 100:.1f}%)"                    
                    )
                    print(
                        f"non-deceptive samples: {total-deceptive} "
                        f"({(total-deceptive)/total*100:.1f}%)"
                    )


def main():
    """Main function to load and process data"""
    loader = DataLoader()

    try:
        print("loading data from HuggingFace...")
        data = loader.load_all_domains(sample_size=200, 
                                       force_reload=True)

        print("\nsaving data...")
        loader.save_processed_data(data)

        print("\ndataset statistics:")
        loader.print_statistics(data)

        return data

    except Exception as e:
        print(f"error in data loading: {str(e)}")
        return None


if __name__ == "__main__":
    main()
