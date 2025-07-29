import click
import warnings
from torch.utils.data import Dataset
def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))