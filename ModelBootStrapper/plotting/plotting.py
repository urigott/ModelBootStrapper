import numpy as np
import pandas as pd

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Whisker, HoverTool, Range1d, Slider, CustomJS
from bokeh.layouts import column
from bokeh.resources import INLINE
from bokeh.embed import file_html


def plot_predict(self, X, y_true=None, threshold=None, samples=50):
    threshold = threshold if threshold else self.threshold

    samples_idx = self._choose_samples_for_plot(X, samples)

    X = X.loc[samples_idx]
    preds = self.predict(X).round(2)

    y_provided = isinstance(y_true, (pd.Series))
    if y_provided:
        preds["y_true"] = y_true.loc[samples_idx]
        preds["colors"] = preds["y_true"].replace({0: "#79DE79", 1: "#FB6962"})

    index_name = preds.index.name if preds.index.name else "index"
    preds = preds.reset_index().sort_values(by="point_est")
    preds["str_index"] = [str(i + 1) for i in range(len(preds))]
    source = ColumnDataSource(data=preds)

    TOOLTIPS, circles_config, whisker_config = self._get_plot_config(
        source=source, y_provided=y_provided, index_name=index_name
    )

    p = figure(width=800, height=400, tools="")
    p.toolbar.logo = None
    circle_glyph = p.circle(**circles_config)
    p.add_layout(Whisker(**whisker_config))

    line_source = ColumnDataSource(
        data={"threshold": [threshold, threshold], "x": [0, len(samples_idx)]}
    )

    p.line(
        x="x",
        y="threshold",
        source=line_source,
        color="black",
        line_dash="dashed",
        line_width=2,
    )
    slider = Slider(
        start=0,
        end=1.0,
        value=threshold,
        step=0.01,
        title="Probability threshold",
    )
    p.yaxis.axis_label = "probability"
    p.y_range = Range1d(0, 1)
    p.xaxis.visible = False

    update_line = CustomJS(
        args=dict(source=line_source, slider=slider),
        code="""const data = source.data;
                const new_y = slider.value;
                data.threshold = [new_y, new_y];
                source.change.emit();
            """,
    )

    slider.js_on_change("value", update_line)

    hover_tool = HoverTool(renderers=[circle_glyph], tooltips=TOOLTIPS)
    p.add_tools(hover_tool)
    p = column(p, slider)
    return file_html(p, INLINE)


@staticmethod
def _get_plot_config(source, y_provided=False, index_name="index"):
    TOOLTIPS = [
        (index_name, f"@{index_name}"),
        (
            "probability",
            "@point_est{0.00} (@lower_bound{0.00} - @upper_bound{0.00})",
        ),
    ]

    circles_config = {
        "source": source,
        "x": "str_index",
        "y": "point_est",
        "size": 7,
    }

    whisker_config = {
        "source": source,
        "base": "str_index",
        "upper": "upper_bound",
        "lower": "lower_bound",
    }

    if y_provided:
        TOOLTIPS.append(("True value", "@y_true"))
        circles_config["color"] = "colors"
        whisker_config["line_color"] = "colors"

    return TOOLTIPS, circles_config, whisker_config


@staticmethod
def _choose_samples_for_plot(X, samples):
    if isinstance(samples, int):
        if samples > len(X):
            raise ValueError(
                "samples must be less or equal to the total number of samples"
            )
        samples_idx = np.random.choice(X.index, size=samples, replace=False)
    elif isinstance(samples, (list, np.ndarray)):
        samples_idx = list(samples)
    elif samples == "all":
        samples_idx = X.index
    else:
        raise ValueError("samples has to be int (number of random samples) or an array")
    return samples_idx
