import argparse
import tqdm
import yaml
from pathlib import Path
from generate_representations import get_representation_function

def get_settings(file_path):
    """Load settings from a YAML file."""
    with open(file_path, 'r') as file:
        settings = yaml.safe_load(file)
    print("Loading settings from YAML:")
    print(f"Using dataset path: {settings.get('dataset_path', 'Not specified')}")
    print(f"Representation type: {settings.get('representation_type', 'Not specified')}")
    print(f"Template ranking: {settings.get('template_ranking', 'Not specified')}")
    return settings


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Run benchmark tests.")
    parser.add_argument("--dataset_path", type=str, help="Optional path to the dataset.")
    parser.add_argument("--representation_type", type=str, default="continuous", help="Type of representation (discrete/continuous) to use.")
    parser.add_argument("--template_ranking", type=str, default="avg", help="Template ranking method (avg/min) to use.")
    args = parser.parse_args()

    if args.dataset_path:
        print(f"Using dataset path: {args.dataset_path}")
    else:
        settings = get_settings(Path("map.yml"))
        dataset_path = settings.get("dataset_path", None)
        if dataset_path is None:
            raise ValueError("Dataset path must be specified either via command line or in map.yaml.")        
        representation_type = settings.get("representation_type", "continuous")
        template_ranking = settings.get("template_ranking", "avg")
        rep_fn_name = settings.get("representation_function", "mhubert")
        rep_fn = get_representation_function(rep_fn_name)

        features = rep_fn(dataset_path)

