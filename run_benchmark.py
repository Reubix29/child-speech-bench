import argparse
import numpy as np
import yaml
from pathlib import Path
from generate_representations import get_representation_function, get_representation_function_type, get_distance_function, initialize_models
from helpers import calculate_metrics

def get_settings(file_path):
    """Load settings from a YAML file."""
    with open(file_path, 'r') as file:
        settings = yaml.safe_load(file)
    print("Loading settings from YAML:")
    print(f"Using dataset path: {settings.get('dataset_path', 'Not specified')}")
    print(f"Representation function: {settings.get('representation_fn', 'Not specified')}")
    print(f"Distance function: {settings.get('distance_fn', 'Not specified')}")
    print(f"Template ranking: {settings.get('template_ranking', 'Not specified')}")
    print(f"Template dirname: {settings.get('template_dirname', 'Not specified')}")

    return settings


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Run benchmark tests.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset.")
    parser.add_argument("--representation_fn", type=str, default="mhubert", help="Name of the representation function to use.")
    parser.add_argument("--distance_fn", type=str, default="dtw", help="Name of the distance function to use.")
    parser.add_argument("--template_ranking", type=str, default="avg", help="Template ranking method (avg/min) to use.")
    parser.add_argument("--template_dirname", type=str, default="sp_1", help="Directory name for templates within the dataset.")
    args = parser.parse_args()

    # Fetch the representation and distance functions
    if args.dataset_path:
        print(f"Using dataset path: {args.dataset_path}")
        dataset_path = Path(args.dataset_path)
        representation_fn_name = args.representation_fn
        distance_fn_name = args.distance_fn
        template_ranking = args.template_ranking
        template_dirname = args.template_dirname
    else:
        settings = get_settings(Path("map.yml"))
        dataset_path = Path(settings.get("dataset_path", None))
        if dataset_path is None:
            raise ValueError("Dataset path must be specified either via command line or in map.yaml.")        
        representation_fn_name = settings.get("representation_fn", "mhubert")
        distance_fn_name = settings.get("distance_fn", "dtw")
        template_ranking = settings.get("template_ranking", "avg")
        template_dirname = settings.get("template_dirname", "sp_1")

    initialize_models(representation_fn_name)
    rep_fn = get_representation_function(representation_fn_name)
    rep_fn_type = get_representation_function_type(representation_fn_name)
    dist_fn = get_distance_function(distance_fn_name)

    if not rep_fn or not dist_fn:
        raise ValueError(f"Invalid representation function '{representation_fn_name}' or distance function '{distance_fn_name}' specified.")

    # Calculate metrics
    metrics = calculate_metrics(
        file_path=dataset_path,
        rep_fn=rep_fn,
        rep_type=rep_fn_type,
        dist_fn=dist_fn,
        template_ranking=template_ranking,
        template_dirname=template_dirname
    )

    print("\nMetrics on the test set:\n==============================")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("==============================")

