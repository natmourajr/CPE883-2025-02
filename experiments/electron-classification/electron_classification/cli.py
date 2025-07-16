from typer import Typer
from .faker import app as faker_app
from .kan import app as kan_app
from .mlp import app as mlp_app

app = Typer()
app.add_typer(faker_app)
app.add_typer(kan_app)
app.add_typer(mlp_app)

if __name__ == "__main__":
    app()
