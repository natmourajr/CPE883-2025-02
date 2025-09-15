"""
Loader for Salesforce/wikitext dataset
"""
from datasets import DatasetDict

from .base import HFDataset


class WikiTextDataset(HFDataset):

    def __init__(
        self,
        cache_dir: str | None = None,
    ):
        super().__init__(
            dataset_name='Salesforce/wikitext',
            dataset_config='wikitext-103-raw-v1',
            text_field='text',
            cache_dir=cache_dir,
        )

    def preprocess(self, ds: DatasetDict) -> DatasetDict:

        return ds.filter(
            lambda x: bool(x[self.text_field].strip())
        )
