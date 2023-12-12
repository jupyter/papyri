import json

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
import tornado

from papyri.crosslink import GraphStore, ingest_dir
from papyri.render import HtmlRenderer, encoder


class RouteHandler(APIHandler):
    # The following decorator should be present on all verb methods (head, get, post,
    # patch, put, delete, options) to ensure only authorized user can request the
    # Jupyter server
    @tornado.web.authenticated
    def post(self):
        body_data = self.request.body.decode()
        g = GraphStore(ingest_dir)
        self.log.error(body_data)
        candidates = [k for k in g.glob((None, None, None, body_data + "*"))]
        the_one = [k for k in candidates if k.path == body_data]

        renderer = HtmlRenderer(g, sidebar=False, prefix="/p/", trailing_html=False)
        if len(the_one) == 1:
            gbytes, backward, forward = g.get_all(the_one[0])
            doc_blob = encoder.decode(gbytes)
            root = renderer._myst_root(doc_blob)
            body = root.to_dict()
        else:
            body = None

        self.finish(
            json.dumps(
                {
                    "data": [c.path for c in candidates[:10]],
                    "body": body,
                }
            )
        )

    @tornado.web.authenticated
    def get(self):
        g = GraphStore(ingest_dir)
        self.finish(
            json.dumps({"data": [k.path for k in g.glob((None, None, None, "p*"))]})
        )


def setup_handlers(web_app):
    host_pattern = ".*$"

    base_url = web_app.settings["base_url"]
    route_pattern = url_path_join(base_url, "papyri-lab", "get-example")
    handlers = [(route_pattern, RouteHandler)]
    web_app.add_handlers(host_pattern, handlers)
