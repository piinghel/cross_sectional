import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def hello_world():
    """"
    """
    print("hello world")

def impute_missing_data(df: pd.DataFrame):
    """
    """
    # forward fill
    df = df.ffill(axis=0)
    # impute mean row
    return df.fillna(df.mean(axis=1))

def make_bins(df: pd.DataFrame, equal_bins: bool, nr_bins: int):
    if equal_bins:
        df += np.random.normal(0, 1e-5, df.shape[0] * df.shape[1]).reshape(df.shape[0], df.shape[1])
        return df.apply(lambda x: pd.qcut(x, nr_bins, labels=False), axis=1) + 1
    else:
        return df.apply(lambda x: pd.cut(x, nr_bins, labels=False), axis=1) + 1

def signal_make_bins(df, nr_bins: int=2, equal_bins: bool=True, impute_missing_values: bool=True) -> pd.DataFrame:
    """
    """
    if nr_bins > df.shape[1]:
        raise ValueError(f"Number of bins (={nr_bins}) should be lower than the number of columns/assets (={df.shape[1]}).")
    if nr_bins < 2:
        raise ValueError(f"Number of bins should at greater than 1.")
    if impute_missing_values:
        df = impute_missing_data(df)
    return make_bins(df, equal_bins, nr_bins)
    
def to_long_format(df:pd.DataFrame, column_name: str="signal_bin") -> pd.DataFrame:
    """
    """
    df.index.name = "date"
    return (df.reset_index()
        .melt(id_vars=df.index.name, value_vars=df.columns)
        .rename(columns={"value": column_name})
        .sort_values(by=[df.index.name, column_name])
        .set_index(df.index.name)
        )

def add_raw_signal(signal_bins: pd.DataFrame, signal_raw: pd.DataFrame, on=["date", "variable"], how="left") -> pd.DataFrame:
    """
    """
    signal_raw_long =  signal_raw.pipe(to_long_format, column_name="signal_raw")
    return signal_bins.reset_index().merge(signal_raw_long.reset_index(), on=on, how=how).set_index("date")

def compute_signal(
    df: pd.DataFrame, 
    nr_bins: int=2,
    equal_bins: bool=True,
    impute_missing_values: bool=True,
    output_format :str="long",
    keep_raw_signal: bool=True) -> pd.DataFrame:   
    """
    """
    signal_wide = signal_make_bins(df, nr_bins, equal_bins, impute_missing_values)
    if output_format == "long":
        signal_long = signal_wide.pipe(to_long_format)
        if keep_raw_signal:
            return signal_long.pipe(add_raw_signal, df).loc[:, ["variable", "signal_raw", "signal_bin"]]
        else:
            return signal_long
    elif output_format == "wide":
        return signal_wide
    else:
        raise ValueError(f"format should be 'wide' or 'long'")
    
def visualize_signal_time(df_wide:pd.DataFrame, title: str="signal/bin", xlab: str="Date", ylab: str="Instruments", height: int=600, width: int=1000, **kwargs):
    """
    """
    fig = go.Figure(data=go.Heatmap(
    colorbar={"title": title},
    z=df_wide.T.values,
    x=df_wide.index.strftime('%Y-%m-%d'),
    y=df_wide.columns))

    fig.update_layout(
        xaxis_title=xlab,
        yaxis_title=ylab,
        height=height,
        width=width,
        **kwargs
        )
    return fig


def merge_ret_signal(ret: pd.DataFrame, signal: pd.DataFrame, on: str="same_date") -> pd.DataFrame:
    """
    """
    ret.index.name = "date"
    ret = ret.copy().reset_index()
    signal.index.name = "date"
    signal = signal.copy().reset_index()
    if on == "same_date":
        return (ret
            .merge(signal, on=["date", "variable"], how="left")
            .sort_values(by=["date", "signal_raw"])
            .set_index("date")
            .loc[:, ["variable", "signal_raw", "signal_bin", "ret"]]
        )
    elif on == "week_year":
        ret["week"] = ret.date.dt.isocalendar().week
        ret["year"] = ret.date.dt.year
        signal["week"] = signal.date.dt.isocalendar().week
        signal["year"] = signal.date.dt.year
        return (ret
            .merge(signal.loc[:, signal.columns != 'date'], on=["week", "year", "variable"], how="right")
            .sort_values(by=["date", "signal_raw"])
            .set_index("date")
            .loc[:, ["variable", "signal_raw", "signal_bin", "ret"]]
        )
    elif on == "month_year":
        ret["month"] = ret.date.dt.month
        ret["year"] = ret.date.dt.year
        signal["month"] = signal.date.dt.month
        signal["year"] = signal.date.dt.year
        return (ret
            .merge(signal.loc[:, signal.columns != 'date'], on=["month", "year", "variable"], how="right")
            .sort_values(by=["date", "signal_raw"])
            .set_index("date")
            .loc[:, ["variable", "signal_raw", "signal_bin", "ret"]]
        )
    else:
        raise ValueError("on can only take values: 'same_date', 'week_year', 'month_year'")

    
def compute_rolling_vol(df:pd.DataFrame, emw=True, **kwargs) -> pd.DataFrame:
    """
    """
    if emw:
        return (df
        .ewm(**kwargs)
        .std()
        )
    else:
        return (df
        .rolling(**kwargs)
        .std()
        )
    


def add_to_df(df1: pd.DataFrame, df2: pd.DataFrame, on: list=["date", "variable"], how: str="left"):
    """
    """
    df1.index.name = "date"
    df2.index.name = "date"
    return df1.reset_index().merge(df2.reset_index(), on=on, how=how)

def normalize_weight_by_group(w, column_name, group, add_to_df=True):
    """
    """
    if add_to_df:
        return (w.groupby(group)[column_name].sum()
            .reset_index()
            .rename(columns={column_name: "allocation"})
            .merge(w, on=group, how="right")
        )
    else:
        return (w.groupby(group)[column_name].sum()
            .reset_index()
            .rename(columns={column_name: "allocation"})
        )


def compute_weigted_ret(ret):
    """
    """
    return (ret
        .assign(w_ret=lambda x: x["ret"] * x["weighting"])
        .groupby(["date", "signal_bin"])["ret"]
        .sum()
        .reset_index()
        .set_index("date")
    )
    
def add_rank(df):
    """
    """
    return df.assign(rank=lambda x: x.groupby("date")["ret"].rank())


def allocation_vol(ret: pd.DataFrame, input_format="wide", **kwargs1) -> pd.DataFrame:
    """
    """
    pass
        


def equal_allocator(df: pd.DataFrame, input_format: str, output_format: str="long") -> pd.DataFrame:
    """
    """
    
    if input_format not in ["long", "wide"]:
        raise ValueError("input_format should be 'long' or 'wide'")
    if output_format not in ["long", "wide"]:
        raise ValueError("output_format should be 'long' or 'wide'")
    df = df.copy()
    if input_format == "wide":
        df = df.pipe(to_long_format, column_name="ret")
    df = df.dropna()
    df["allocation"] = 1 / df.groupby("date")["ret"].count()
    if output_format == "long":
        return df  
    elif output_format == "wide":
        return df.reset_index().pivot(index="date", columns="variable", values="allocation")
    


def vol_alloctor(df: pd.DataFrame, group: str="date", **kwargs):
    """
    """
    pass

def group_alloctor(df: pd.DataFrame, group: str="date", **kwargs):
    """
    """
    pass

def create_allocation():
    """
    """
    pass
    


def compute_return_series_bin(ret_sig: pd.DataFrame, how: str="equal", ret_col: str="ret", signal_col: str="signal_bin", output_format="long", **kwargs):
    """
    """
    
    ret_sig.index.name = "date"
    if how == "equal":
        out = (ret_sig.reset_index().groupby(["date", signal_col], as_index=True)[[ret_col]]
            .mean()
            .reset_index()
            .set_index("date")
            )

    elif how == "vol":
        ret_wide = ret_sig.reset_index().pivot(index="date", columns="variable", values="ret")
        out = (ret_wide
            .pipe(compute_rolling_vol, **kwargs)
            .pipe(to_long_format, column_name="vol")
            .pipe(add_to_df, ret_sig, on=["date", "variable"], how="right")
            .pipe(normalize_weight_by_group, column_name="vol", group=["date", signal_col], add_to_df=True)
            .pipe(compute_weigted_ret)
            )
    elif how == "rank":
        pass
    else:
        raise ValueError("weighting can only take equal, vol, rank")
    
    if output_format == "long":
        return out
    elif output_format == "wide":
        return out.reset_index().pivot(index="date", columns="signal_bin", values="ret")


def compute_cumsum(df: pd.DataFrame, input_format: str=None, output_format: str="long") -> pd.DataFrame:
    """
    """
    if input_format is None:
        raise ValueError("input format shold be long or wide")
    if input_format == "long":
        df = df.reset_index().pivot(index="date", columns="signal_bin", values="ret")
    elif input_format == "wide":
        pass
    else:
        raise ValueError("input_format can only take long or wide")
    df_cumsum = (1 + df.cumsum()).fillna(method="ffill")
    if output_format == "long":
        return (df_cumsum
        .reset_index()
        .melt(id_vars="date", value_vars=df.columns, var_name="signal_bin", value_name="cum_ret")
        .sort_values(by=["date", "signal_bin"])
        .set_index("date")
        )
    elif output_format == "wide":
        return df_cumsum
    else:
        raise ValueError("output_format can only take long or wide")

    
def visualize_returns_series_bin(
    ret: pd.DataFrame,
    input_format: str=None,
    target_vol: float=0.05/np.sqrt(52),
    x: str="date", 
    y: str="cum_ret", 
    color: str="signal_bin", 
    xlab: str="Date", 
    ylab: str="Cum sum return",
    height: int=600,
    width: int=800, 
    **kwargs):
    """
    """
    
    if input_format is None:
        raise ValueError("input format shold be long or wide")
    if input_format == "long":
        ret = ret.reset_index().pivot(index="date", columns="signal_bin", values="ret")   
    if target_vol > 0:
        ret = ret * target_vol / ret.std()
    cum_ret_long = (ret
        .pipe(compute_cumsum, input_format="wide", output_format="long")
        .reset_index()
        )
    fig = px.line(
        cum_ret_long.reset_index(),
        x=x,
        y=y,
        color=color,
        height=height,
        width=width
    )
    fig.update_layout(
        xaxis_title=xlab,
        yaxis_title=ylab,
        **kwargs
        )
    return fig

def make_barplot(df):
    """
    """
    return (
        px.bar(
            df,
            x="date",
            y="ret",
            color="signal_bin",
            barmode="group"
        )
    )
    

def compute_performance_statistics(df: pd.DataFrame, func_dict):
    """
    """
    store_stats = []
    for k, v in func_dict.items():
        out = df.groupby("signal_bin")["ret"].apply(v).to_frame().reset_index()
        out["Statistic"] = k
        store_stats.append(out)
    perf = pd.concat(store_stats, axis=0).rename(columns={"ret": "Value", "signal_bin": "Bin"})
    perf["Bin"] = perf["Bin"].astype("category")
    return perf 


def plot_performance_statistics(
    df: pd.DataFrame,
    x: str="Bin",
    y: str="Value",
    color: str="Statistic",
    height: int=800, 
    width: int=800,
    **kwargs):
    """
    """ 
    fig = px.bar(
        df, 
        x=x, 
        y=y, 
        facet_col=color, 
        facet_col_wrap=2, 
        color=color, 
        facet_row_spacing=0.075,
        facet_col_spacing=0.075,
        height=height,
        width=width,
    )

    nr_bin = len(df[x].unique()) + 1
    fig.for_each_annotation(lambda x: x.update(text=f"{x.text.split('=')[-1]}"))
    fig.update_yaxes(matches=None)
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True, nticks=4))
    fig.update_xaxes(matches=None)
    fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True, nticks=nr_bin))
    fig.update_layout(
        showlegend=False,
        autosize=False,
        **kwargs
    )
    return fig 

def compute_long_short_portfolio(df, long_bin :list, short_bin: list, weight_func, **kwargs):
    """
    """   
    df_long = df[df['signal_bin'].isin(long_bin)]
    df_short = df[df['signal_bin'].isin(short_bin)]
    # apply weighting
    df_long = df_long.groupby(df_long.index).apply(weight_func)
    df_short = df_short.groupby(df_short.index).apply(weight_func)
    df_long_short = df_long - df_short
    df_long["signal_bin"] = "long"
    df_short["signal_bin"] = "short"
    df_long_short["signal_bin"] = "long_short"    
    return pd.concat([df_long, df_short, df_long_short], axis=0)

    
            
