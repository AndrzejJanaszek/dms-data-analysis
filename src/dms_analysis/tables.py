import pandas as pd
from IPython.display import HTML


def collapsible(frame: pd.DataFrame, title: str, opened: bool = False) -> HTML:
    state = " open" if opened else ""
    return HTML(f"<details{state}><summary><b>{title}</b> ({len(frame)} wierszy)</summary>{frame.to_html()}</details>")
