from io import StringIO

from pandas import DataFrame, read_html


def html_to_df(html: str, index=0, /, **kwargs) -> DataFrame:
    return read_html(StringIO(html), **kwargs)[index]
