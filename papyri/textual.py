"""
papyri textual
"""

from pathlib import Path

from textual.app import App, ComposeResult, RenderResult
from textual import events
from textual.binding import Binding
from textual.containers import VerticalScroll, Container
from textual.widgets import Header, Footer, Label, Static

from emoji import emojize

from papyri.crosslink import encoder
from papyri.config import ingest_dir


class Signature(Label):
    DEFAULT_CSS = """
    Signature {
        width: 100%;
        border: solid white;
        padding-left: 3;
        padding-top: 1;
        padding-bottom: 1;
    }
    """

    def __init__(self, qualname, signature):
        super().__init__()
        self.qualname = qualname
        self.signature = self.extract_Signature(signature)

    def extract_Annotation(self, annotation):
        if "Empty" in str(annotation):
            ret_annotation = "None"
        else:
            ret_annotation = ("signature", annotation)
        return ret_annotation

    def extract_Parameters(self, parameters):
        return ", ".join([f"{p.name}" for p in parameters])

    def extract_Signature(self, sig):
        if sig:
            kind = sig.kind
            parameters = self.extract_Parameters(sig.parameters)
            return_annotation = self.extract_Annotation(sig.return_annotation)
            return f"{kind} {self.qualname.replace(':', '.')}({parameters}) -> {return_annotation}"
        else:
            return f"{self.qualname.replace(':', '.')}()"

    def render(self) -> RenderResult:
        return self.signature


class Body(Static):
    def __init__(self, item):
        super().__init__()
        self.item = item

    def render(self) -> RenderResult:
        return self.item


class PapyriApp(App):
    TITLE = emojize("papyri :palm_tree:")

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit the app"),
    ]
    CSS_PATH = Path("static/papyri.tcss")

    def run(self, name, **kwargs):
        from papyri.graphstore import GraphStore
        from papyri.render import _rich_render
        store = GraphStore(ingest_dir, {})
        key = next(iter(store.glob((None, None, "module", name))))

        self.things =  _rich_render(key, store)
        super().run(**kwargs)

    def compose(self) -> ComposeResult:
        """
        Renders the layout on screen.
        """
        from rich.console import Group

        #if self.blob.signature:
        #    signature = self.blob.signature
        #else:
        #    signature = None
        # content = str(self.blob.content)

        yield Container(
            Header(),
            #Signature(qualname=self.qualname, signature=signature),
            VerticalScroll(
                *[Body(t) for t in self.things]
            ),
            Footer(),
        )

    def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.exit()


def load(file_path):
    blob = encoder.decode(file_path.read_bytes())
    assert hasattr(blob, "arbitrary")
    return blob


def guess_load(qualname):
    """
    Try to load JSON file for this object
    """
    candidates = list(ingest_dir.glob(f"*/*/module/{qualname}"))
    if candidates:
        for file_path in candidates:
            try:
                blob = load(file_path)
            except Exception as e:
                raise ValueError(str(file_path)) from e
    return blob


def main(qualname: str):

    app = PapyriApp()
    app.run(qualname)


def setup() -> None:
    import sys

    target: str
    target = sys.argv[1]
    assert isinstance(target, str)
    res = main(target)
    print(res)


if "__main__" == __name__:
    setup()
