"""
Loader for original 20 newsgroups dataset
"""
import re

from datasets import DatasetDict

from .base import HFDataset


class TwentyNewsgroupsDataset(HFDataset):

    def __init__(
        self,
        cache_dir: str | None = None,
    ):
        super().__init__(
            dataset_name="SetFit/20_newsgroups",
            dataset_config='default',
            text_field='text',
            cache_dir=cache_dir,
        )

        self.url_re = re.compile(r'https?://[^\s]+')

    def preprocess(self, ds: DatasetDict) -> DatasetDict:

        return (
            ds
            .filter(lambda x: bool(x[self.text_field].strip()))
            .map(lambda x: {
                self.text_field: self.url_re.sub('', x[self.text_field])
            })
        )

