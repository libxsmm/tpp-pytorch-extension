import argparse
import dgl
import torch
import json
import os
import os.path as osp
from dgl.distributed.graph_partition_book import (
    _etype_str_to_tuple,
    _etype_tuple_to_str,
)
import dgl.backend as F
from dgl.base import EID, ETYPE, NID, NTYPE
from dgl.data.utils import save_graphs

RESERVED_FIELD_DTYPE = {
    "inner_node": (
        F.uint8
    ),  # A flag indicates whether the node is inside a partition.
    "inner_edge": (
        F.uint8
    ),  # A flag indicates whether the edge is inside a partition.
    NID: F.int64,
    EID: F.int64,
    NTYPE: F.int16,
    # `sort_csr_by_tag` and `sort_csc_by_tag` works on int32/64 only.
    ETYPE: F.int32,
}

def _format_part_metadata(part_metadata, formatter):
    """Format etypes with specified formatter."""
    for key in ["edge_map", "etypes"]:
        if key not in part_metadata:
            continue
        orig_data = part_metadata[key]
        if not isinstance(orig_data, dict):
            continue
        new_data = {}
        for etype, data in orig_data.items():
            etype = formatter(etype)
            new_data[etype] = data
        part_metadata[key] = new_data
    return part_metadata


def _load_part_config(part_config):
    """Load part config and format."""
    try:
        with open(part_config) as f:
            part_metadata = _format_part_metadata(
                json.load(f), _etype_str_to_tuple
            )
    except AssertionError as e:
        raise DGLError(
            f"Failed to load partition config due to {e}. "
            "Probably caused by outdated config. If so, please refer to "
            "https://github.com/dmlc/dgl/tree/master/tools#change-edge-"
            "type-to-canonical-edge-type-for-partition-configuration-json"
        )
    return part_metadata


def _dump_part_config(part_config, part_metadata):
    """Format and dump part config."""
    part_metadata = _format_part_metadata(part_metadata, _etype_tuple_to_str)
    with open(part_config, "w") as outfile:
        json.dump(part_metadata, outfile, sort_keys=False, indent=4)

def _save_graphs(filename, g_list, formats=None, sort_etypes=False):
    """Preprocess partitions before saving:
    1. format data types.
    2. sort csc/csr by tag.
    """
    for g in g_list:
        for k, dtype in RESERVED_FIELD_DTYPE.items():
            if k in g.ndata:
                g.ndata[k] = F.astype(g.ndata[k], dtype)
            if k in g.edata:
                g.edata[k] = F.astype(g.edata[k], dtype)
    for g in g_list:
        if (not sort_etypes) or (formats is None):
            continue
        if "csr" in formats:
            g = sort_csr_by_tag(g, tag=g.edata[ETYPE], tag_type="edge")
        if "csc" in formats:
            g = sort_csc_by_tag(g, tag=g.edata[ETYPE], tag_type="edge")
    save_graphs(filename, g_list, formats=formats)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Post-Process Partitions")
    argparser.add_argument(
        "--part_path",
        type=str,
        help="dataset partitions path",
    )
    argparser.add_argument(
        "--token",
        type=str,
        default="p",
        help="token identify partitions path",
    )
    argparser.add_argument(
        "--num_parts",
        type=int,
        help="number of partitions",
    )
   
    args = argparser.parse_args()

    part_md = []
    for p in range(args.num_parts):
        pname = osp.join(args.part_path+args.token, 'part') + str(p)
        pname = osp.join(pname, 'IGBH.json')
        part_md.append(_load_part_config(pname))

    pm = part_md[0]
    nm = pm['node_map']
    nkeys = nm.keys()
    em = pm['edge_map']
    ekeys = em.keys()

    for k in nkeys:
        fnm = []
        fnm.append(nm[k])
        for p in range(1, args.num_parts):
            tpm = part_md[p]
            tnm = tpm['node_map']
            fnm.append(tnm[k])
        nm[k] = fnm
    pm['node_map'] = nm

    for k in ekeys:
        fem = []
        fem.append(em[k])
        for p in range(1, args.num_parts):
            tpm = part_md[p]
            tem = tpm['edge_map']
            fem.append(tem[k])
        em[k] = fem
    pm['edge_map'] = em

    out_path = osp.join(args.part_path + args.token, 'IGBH.json')
    _dump_part_config(out_path, pm)

    for p in range(args.num_parts):
        gpath = osp.join(args.part_path + args.token, 'part') + str(p)
        gpath = osp.join(gpath, 'graph.pt')
        g = torch.load(gpath)
        ogpath = osp.join(args.part_path + args.token, 'part') + str(p)
        ogpath = osp.join(ogpath, 'graph.dgl')
        _save_graphs(ogpath, [g])
        cmd = "rm " + gpath
        os.system(cmd)

        pname = osp.join(args.part_path + args.token, 'part') + str(p)
        pname = osp.join(pname, 'IGBH.json')
        cmd = "rm " + pname
        os.system(cmd)

