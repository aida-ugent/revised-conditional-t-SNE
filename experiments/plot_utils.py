import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

sns.set_theme(
    context="notebook",
    font_scale=1.5,
    style="ticks",
    rc={"figure.dpi": 100, "axes.spines.right": False, "axes.spines.top": False},
)


def plot_embedding(
    embedding_df, color_col, shape_col, ax=None, title=None, palette=None, markers=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    g = sns.scatterplot(
        data=embedding_df,
        x=embedding_df["x"],
        y=embedding_df["y"],
        alpha=0.5,
        hue=embedding_df[color_col],
        linewidth=0.5,
        style=shape_col,
        markers=markers if markers is not None else True,
        palette=palette,
        legend=None,
        ax=ax,
        s=20,
    )
    ax.set_aspect("equal")
    # ax.legend().remove()
    if title is not None:
        ax.set_title(f"{title}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    return g


def plot_cell_embedding(
    embedding_df, color_col, ax=None, title=None, palette=None, legend=False
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if palette is not None:
        vals_unique, vals_counts = np.unique(
            embedding_df[color_col], return_counts=True
        )
        vals_unique = [vals_unique[i] for i in np.argsort(vals_counts)[::-1]]

        for v in vals_unique:
            g = sns.scatterplot(
                data=embedding_df.loc[embedding_df[color_col] == v],
                x=embedding_df["x"],
                y=embedding_df["y"],
                alpha=0.5,
                hue=color_col,
                linewidth=0,
                palette=palette,
                ax=ax,
                s=15,
            )
    else:
        g = sns.scatterplot(
            data=embedding_df,
            x=embedding_df["x"],
            y=embedding_df["y"],
            alpha=0.5,
            hue=color_col,
            linewidth=0,
            ax=ax,
            s=15,
        )
    ax.set_aspect("equal")
    if not legend:
        ax.legend().remove()
    if title is not None:
        ax.set_title(f"{title}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    return ax


def plotly_scatter(embedding_df, color_col, shape_col, hover_cols=None, colormap=None):
    fig = px.scatter(
        embedding_df,
        x="x",
        y="y",
        color=color_col,
        symbol=shape_col,
        hover_data=hover_cols,
        template="simple_white",
        opacity=0.5,
        width=1000,
        height=1000,
        color_discrete_map=colormap,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        visible=False,
    )
    fig.update_xaxes(visible=False)
    return fig
