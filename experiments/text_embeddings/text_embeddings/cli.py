from pathlib import Path
from rich import print
from typer import Typer, Exit

from . import tokenizer
from . import embeddings
from . import embedding_generator

import hf_dataset_loader

app = Typer()


@app.command()
def extract_hf_bert_embeddings(
    model_name: str,
    model_path: Path,
    output_path: Path | None = None,
    batch_size: int = 32,
    bos_token: str | None = None,
    eos_token: str | None = None,
    mask_token: str | None = None,
):

    if not (model_path.exists() and model_path.is_dir()):
        print(f"Model path {model_path} does not exist or is not a directory.")
        #raise Exit(code=1)

    tokenizer_kwargs = {}
    if bos_token is not None:
        tokenizer_kwargs['bos_token'] = bos_token
    if eos_token is not None:
        tokenizer_kwargs['eos_token'] = eos_token
    if mask_token is not None:
        tokenizer_kwargs['mask_token'] = mask_token

    extractor = embeddings.HFBertExtractor(str(model_path), **tokenizer_kwargs)
    extractor.load_model()

    if output_path is None:
        output_path = Path(f"data/embeddings/{model_name.replace('/', '_')}_embeddings.h5")

    print(f'Saving to {output_path}')

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    loader = hf_dataset_loader.TwentyNewsgroupsDataset()

    print("Extracting embeddings...")
    embedding_generator.extract_embeddings_to_hdf5(
        loader=loader,
        extractor=extractor,
        model_name=model_name,
        output_path=str(output_path),
        batch_size=batch_size,
    )


@app.command()
def extract_composer_embeddings(
    model_name: str,
    model_path: Path,
    output_path: Path | None = None,
    batch_size: int = 32,
    bos_token: str | None = None,
    eos_token: str | None = None,
    mask_token: str | None = None,
):

    if not (model_path.exists() and model_path.is_file()):
        print(f"Model path {model_path} does not exist or is not a directory.")
        raise Exit(code=1)

    tokenizer_kwargs = {}
    if bos_token is not None:
        tokenizer_kwargs['bos_token'] = bos_token
    if eos_token is not None:
        tokenizer_kwargs['eos_token'] = eos_token
    if mask_token is not None:
        tokenizer_kwargs['mask_token'] = mask_token

    from .embeddings_modernbert import ComposerExtractor
    extractor =ComposerExtractor(str(model_path), **tokenizer_kwargs)
    extractor.load_model()

    if output_path is None:
        output_path = Path(f"data/embeddings/{model_name.replace('/', '_')}_embeddings.h5")

    print(f'Saving to {output_path}')

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    loader = hf_dataset_loader.TwentyNewsgroupsDataset()

    print("Extracting embeddings...")
    embedding_generator.extract_embeddings_to_hdf5(
        loader=loader,
        extractor=extractor,
        model_name=model_name,
        output_path=str(output_path),
        batch_size=batch_size,
    )


@app.command()
def train_tokenizer(
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    output_dir: Path = Path('data/tokenizer')
):

    print("Training...")
    tok = tokenizer.train_from_hf_dataset(
        name=dataset_name,
        config=dataset_config,
    )

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    print("Saving...")
    tok.save(str(output_dir / 'tokenizer.json'))


if __name__ == "__main__":
    app()
