# Template Loader

Template para Data Loader. Quando quiser criar um novo Data Loader, duplique o conteúdo desse diretório, e lembre-se de mexer nos seguintes pontos:

- Crie um README.md adequado ao Data Loader, explicando o tipo de dados a ser lido
- Edite o arquivo `pyproject.toml`, e realize as seguintes mudanças:
    - Altere o `name` para o nome do data loader ou modelo. É importante que esse nome seja um [identificador válido em Python](https://docs.python.org/3/reference/lexical_analysis.html#identifiers), de preferência seguindo a [convenção de nomes para módulos e pacotes](https://peps.python.org/pep-0008/#package-and-module-names)
    - Ajuste os campos `version`, `description` conforme necessário
    - Ajuste a versão do python necessária em `requires-python`. Aqui, dê preferência a especificações com >=, tal como `>=3.10`.
- Renomeie a pasta `template_loader` para o mesmo nome usado como valor de `name` no `pyproject.toml`.

## Execução de testes

Caso esteja usando uv, basta executar da seguinte forma:

```bash
uv run --extra dev pytest
```

Caso esteja usando somente o `pip` básico para gerencia pacotes, crie um virtualenv e execute os seguintes passos:

```bash
pip install -e '.[dev]'
pytest
```

Para uma descrição mais detalhada sobre `uv` e `pip`, veja o [README do TemplateExperiment](../../experiments/TemplateExperiment/README.md).
