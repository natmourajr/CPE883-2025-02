"""
Loader for wikitext dataset

ds2_name = "zeroshot/twitter-financial-news-topic"
ds2_config = "default"
ds2 = datasets.load_dataset(ds2_name, ds2_config)
"""
import re

from datasets import DatasetDict

from .base import HFDataset


class TwitterFinancialNewsTopicDataset(HFDataset):

    def __init__(
        self,
        cache_dir: str | None = None,
    ):
        super().__init__(
            dataset_name="zeroshot/twitter-financial-news-topic",
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
