import ogbench

dataset_names = [
    'humanoidmaze-medium-navigate-v0',
]
ogbench.download_datasets(
    dataset_names,  # List of dataset names.
    dataset_dir='~/.ogbench/data',  # Directory to save datasets (optional).
)