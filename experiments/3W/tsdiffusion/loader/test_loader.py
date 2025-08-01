# Pytest file for testing the TemplateLoader

from __init__ import Loader3W


def test_template_dataset():
    ld = Loader3W()
    dataset = ld.load_real_instance()
    assert len(dataset) == 10772