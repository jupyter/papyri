import dataclasses
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from collections import defaultdict

from velin import NumpyDocString

from numpydoc.docscrape import Parameter

from .config import base_dir, cache_dir, ingest_dir
from .gen import keepref, normalise_ref
from .take2 import Paragraph
from .utils import progress

import warnings
warnings.simplefilter('ignore', UserWarning)


def resolve_(qa, known_refs, local_ref):
    def resolve(ref):
        if ref in local_ref:
            return ref, "local"
        if ref in known_refs:
            return ref, "exists"
        else:
            parts = qa.split(".")
            for i in range(len(parts)):
                attempt = ".".join(parts[:i]) + "." + ref
                if attempt in known_refs:
                    return attempt, "exists"

        q0 = qa.split(".")[0]
        attempts = [q for q in known_refs if q.startswith(q0) and (ref in q)]
        if len(attempts) == 1:
            return attempts[0], "exists"
        return ref, "missing"

    return resolve


@dataclass
class Ref:
    name: str
    ref: str
    exists: bool

    def __hash__(self):
        return hash((self.name, self.ref, self.exists))

@dataclass
class SeeAlsoItem:
    name: Ref
    descriptions: str
    # there are a few case when the lhs is `:func:something`... in scipy.
    type: str

    @classmethod
    def from_json(cls, name, descriptions, type):
        return cls(Ref(**name), descriptions, type)

    def __hash__(self):
        return hash((self.name, tuple(self.descriptions)))


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def assert_normalized(ref):
    # nref = normalise_ref(ref)
    # assert nref == ref, f"{nref} != {ref}"
    return ref


def load_one(bytes_, bytes2_,  qa=None):
    # blob.see_also = [SeeAlsoItem.from_json(**x) for x in data.pop("see_also", [])]


    data = json.loads(bytes_)
    blob = NumpyDocString.from_json(data)
    #blob._parsed_data = data.pop("_parsed_data")
    data.pop("_parsed_data")
    data.pop("edata", None)
    #blob._parsed_data["Parameters"] = [
    #    Parameter(a, b, c) for (a, b, c) in blob._parsed_data["Parameters"]
    #]
    blob.refs = data.pop("refs", [])
    #blob.edata = data.pop("edata")
    if bytes2_ is not None:
        backrefs = json.loads(bytes2_)
    else:
        backrefs = []
    blob.backrefs = backrefs
    #blob.see_also = data.pop("see_also", [])
    blob.see_also = [SeeAlsoItem.from_json(**x) for x in data.pop("see_also", [])]
    blob.version = data.pop("version", '')
    data.pop('ordered_sections', None)
    data.pop('backrefs', None)
    if data.keys():
        print(data.keys(), qa)
    blob.__dict__.update(data)
    try:
        if (see_also := blob["See Also"]) and not blob.see_also:
            for nts, d in see_also:
                for (n, t) in nts:
                    if t and not d:
                        d, t = t, None
                    blob.see_also.append(
                        SeeAlsoItem(Ref(n, None, None), d, t)
                    )
    except Exception as e:
        raise ValueError(f"Error {qa}: {see_also} | {nts}") from e
    assert isinstance(blob.see_also, list), f"{blob.see_also=}"
    for l in blob.see_also:
        assert isinstance(l, SeeAlsoItem), f"{blob.see_also=}"
    blob.see_also = list(set(blob.see_also))
    try:
        notes = blob["Notes"]
        blob.refs.extend(refs := Paragraph.parse_lines(notes).references)
        # if refs:
        # print(qa, refs)
    except KeyError:
        pass
    blob.refs = list(sorted(set(blob.refs)))
    return blob


def main(name, check):

    if name == 'all':
        name_glob = '**'
    else:
        name_glob = name

    nvisited_items = {}
    versions = {}
    for console, path in progress(cache_dir.glob(f"**/__papyri__.json"), description="Loading package versions..."):
        with path.open() as f:
            version = json.loads(f.read())['version']
        versions[path.parent.name] = version
    for p, f in progress(cache_dir.glob(f"{name_glob}/*.json"), description=f"Reading {name} doc bundle files ..."):
        if f.is_dir():
            continue
        fname = f.name
        if fname == '__papyri__.json':
            continue
        qa = fname[:-5]
        if check:
            rqa = normalise_ref(qa)
            if rqa != qa:
                # numpy weird thing
                print(f'skip {qa}')
                continue
            assert rqa == qa, f"{rqa} !+ {qa}"
        try:
            with f.open() as fff:
                from pathlib import Path
                brpath = Path(str(f)[:-5]+'br')
                if brpath.exists():
                    br = brpath.read_text()
                else:
                    br = None
                blob = load_one(fff.read(), br, qa=qa)
                if check:
                    blob.refs = [assert_normalized(ref) for ref in blob.refs if keepref(ref)]
                nvisited_items[qa] = blob

        except Exception as e:
            raise RuntimeError(f"error Reading to {f}") from e

    for p, (qa, ndoc) in progress(
        nvisited_items.items(), description="Cross referencing"
    ):
        local_ref = [x[0] for x in ndoc["Parameters"] if x[0]]+[x[0] for x in ndoc["Returns"] if x[0]]
        for ref in ndoc.refs:
            resolved, exists = resolve_(qa, nvisited_items, local_ref)(ref)
            # here need to check and load the new files touched.
            if resolved in nvisited_items and ref != qa:
                #print(qa, 'incommon')
                nvisited_items[resolved].backrefs.append(qa)
            elif ref != qa and exists == 'missing':
                r = resolved.split('.')[0]
                ex = ingest_dir / (resolved + '.json')
                if ex.exists():
                    with ex.open() as f:
                        brpath = Path(str(ex)[:-5]+'br')
                        if brpath.exists():
                            br = brpath.read_text()
                        else:
                            br = None
                        blob = load_one(f.read(), br)
                        nvisited_items[resolved] = blob
                        if not hasattr(nvisited_items[resolved], 'backrefs'):
                            nvisited_items[resolved].backrefs = []
                        nvisited_items[resolved].backrefs.append(qa)
                elif '/' not in resolved:
                    phantom_dir = (ingest_dir / '__phantom__')
                    phantom_dir.mkdir(exist_ok=True)
                    ph = phantom_dir / (resolved + '.json')
                    if ph.exists():
                        with ph.open() as f:
                            ph_data = json.loads(f.read())

                    else:
                        ph_data = []
                    ph_data.append(qa)
                    with ph.open('w') as f:
                        #print('updating phantom data', ph)
                        f.write(json.dumps(ph_data))
                else:
                    print(resolved, 'not valid reference, skipping.')



        for sa in ndoc.see_also:
            resolved, exists = resolve_(qa, nvisited_items, [])(sa.name.name)
            if exists == "exists":
                sa.name.exists = True
                sa.name.ref = resolved
    for p, (qa, ndoc) in progress(nvisited_items.items(), description="Cleaning double references"):
        # TODO: load backrref from either:
        # 1) previsous version fo the file
        # 2) phantom file if first import (and remove the phantom file?)
        phantom_dir = (ingest_dir / '__phantom__')
        ph = phantom_dir / (qa + '.json')
        #print('ph?', ph)
        if ph.exists():
            with ph.open() as f:
                ph_data = json.loads(f.read())
            print('loading data from phantom file !', ph_data)
        else:
            ph_data = []

        ndoc.backrefs = list(sorted(set(ndoc.backrefs + ph_data)))
    if name_glob != '**':
        gg = f'{name}*.json'
    else:
        gg = '*.json'
    for console, path in progress(
        ingest_dir.glob(gg), description=f"cleanig previsous files ...."
    ):
        path.unlink()

    for p, (qa, ndoc) in progress(nvisited_items.items(), description="Writing..."):
        root = qa.split('.')[0]
        ndoc.version = versions.get(root, '?')
        js = ndoc.to_json()
        br = js.pop('backrefs', [])
        try:
            path = ingest_dir / f"{qa}.json"
            path_br = ingest_dir / f"{qa}.br"

            with path.open("w") as f:
                f.write(json.dumps(js, cls=EnhancedJSONEncoder))
            if path_br.exists():
                with path_br.open("r") as f:
                    bb = json.loads(f.read())
            else:
                bb= []
            with path_br.open("w") as f:
                f.write(json.dumps(list(sorted(set(br+bb)))))
        except Exception as e:
            raise RuntimeError(f"error writing to {fname}") from e
