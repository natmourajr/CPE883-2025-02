# from typer import Typer

# app = Typer()


# @app.command()
# def display_batches(
#     batch_size: int = 4,
#     shuffle: bool = True,
#     max_batches: int = 10,
# ):
#     """Display batches of the template dataset"""

#     dataset = TemplateDataset()
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     for batch_idx, batch in enumerate(dataloader):
#         print(f"Batch {batch_idx}: {batch}")
#         if batch_idx >= max_batches:
#             break


# if __name__ == "__main__":
#     app()