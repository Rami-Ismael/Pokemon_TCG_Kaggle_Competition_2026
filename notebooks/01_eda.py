import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from pokemon_tcg import config, data

    return config, data, mo


@app.cell
def _(config, mo):
    mo.md(f"# Pokemon TCG — EDA\n\nRaw data directory: `{config.RAW_DATA_DIR}`")
    return


@app.cell
def _(config, data):
    # Uncomment once competition data is downloaded into data/raw/.
    # df = data.load_csv("train.csv")
    # df.head()
    return


if __name__ == "__main__":
    app.run()
