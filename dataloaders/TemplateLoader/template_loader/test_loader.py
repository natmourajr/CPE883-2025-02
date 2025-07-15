# Pytest file for testing the TemplateLoader

from template_loader.loader import TemplateDataset


def test_template_dataset():
    dataset = TemplateDataset()
    assert len(dataset) == 5
    assert dataset[0] == 1
    assert dataset[1] == 2
    assert dataset[2] == 3
    assert dataset[3] == 4
    assert dataset[4] == 5