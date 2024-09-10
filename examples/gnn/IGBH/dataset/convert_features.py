import torch
import argparse
import os.path as osp
import numpy as np
from tpp_pytorch_extension.gnn.common import gnn_utils

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
            "Down-Convert Dataset Node Features"
    )
    argparser.add_argument(
        "--path",
        type=str,
        help="dataset features path",
    )
    argparser.add_argument(
        "--dataset_size",
        type=str,
        help="dataset size",
    )
    argparser.add_argument(
        "--target_dtype",
        type=str,
        help="target data type",
    )
    argparser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="block size for quantization",
    )
    argparser.add_argument(
        "--ntype",
        type=str,
        default='all',
        help="node type",
    )

    args = argparser.parse_args()

    ntypes = ['author', 'fos', 'institute', 'conference', 'journal', 'paper']
    if args.dataset_size in ['large', 'full']:
        if args.target_dtype in ["hf8", "bf8", "bf16"]:
            dt = args.target_dtype
            if args.ntype == 'all':
                for ntype in ntypes:
                    in_name = osp.join(args.path, args.dataset_size, 'processed', ntype, 'node_feat.npy')
                    out_name = osp.join(args.path, args.dataset_size,  'processed', ntype, 'node_feat_'+dt+'.pt')

                    if ntype in ['author', 'paper']:
                        gnn_utils.downconvert_dataset(in_name, out_name, args.target_dtype, 1024)
                    else:
                        if args.target_dtype == "bf16":
                            nf = torch.from_numpy(np.load(in_name)).to(torch.bfloat16)
                        elif args.target_dtype == "hf8":
                            nf = torch.from_numpy(np.load(in_name)).to(torch.float8_e4m3fn)
                        elif args.target_dtype == "bf8":
                            nf = torch.from_numpy(np.load(in_name)).to(torch.float8_e5m2)
                        torch.save(nf, out_name)
            else:
                in_name = osp.join(args.path, args.dataset_size, 'processed', args.ntype, 'node_feat.npy')
                out_name = osp.join(args.path, args.dataset_size,  'processed', args.ntype, 'node_feat_'+dt+'.pt')
                if args.ntype in ['author', 'paper']:
                    gnn_utils.downconvert_dataset(in_name, out_name, args.target_dtype, 1024)
                else:
                    if args.target_dtype == "bf16":
                        nf = torch.from_numpy(np.load(in_name)).to(torch.bfloat16)
                    elif args.target_dtype == "hf8":
                        nf = torch.from_numpy(np.load(in_name)).to(torch.float8_e4m3fn)
                    elif args.target_dtype == "bf8":
                        nf = torch.from_numpy(np.load(in_name)).to(torch.float8_e5m2)
                    torch.save(nf, out_name)
        elif args.target_dtype == "int8":
            bsz = args.block_size
            if args.ntype == 'all':
                for ntype in ntypes:
                    in_name = osp.join(args.path, args.dataset_size, 'processed', ntype, 'node_feat.npy')
                    out_name = osp.join(args.path, args.dataset_size, 'processed', ntype, 'node_feat_int8.pt')
                    scf_name = osp.join(args.path, args.dataset_size, 'processed', ntype, 'node_feat_scf.pt')

                    print(f'Quantizing {ntype} to int8')

                    if ntype in ['author', 'paper']:
                        gnn_utils.quantize_dataset(in_name, out_name, scf_name, 1024, bsz)
                    else:
                        nf = torch.from_numpy(np.load(in_name))
                        gnn_utils.quantize_dataset_feat(nf, out_name, scf_name, int(bsz))
            else:
                in_name = osp.join(args.path, args.dataset_size, 'processed', args.ntype, 'node_feat.npy')
                out_name = osp.join(args.path, args.dataset_size, 'processed', args.ntype, 'node_feat_int8.pt')
                scf_name = osp.join(args.path, args.dataset_size, 'processed', args.ntype, 'node_feat_scf.pt')
                print(f'Quantizing {args.ntype} to int8')

                if args.ntype in ['author', 'paper']:
                    gnn_utils.quantize_dataset(in_name, out_name, scf_name, 1024, bsz)
                else:
                    nf = torch.from_numpy(np.load(in_name))
                    gnn_utils.quantize_dataset_feat(nf, out_name, scf_name, int(bsz))

    else:
        if args.target_dtype == "bf16":
            for ntype in ntypes:
                in_name = osp.join(args.path, args.dataset_size, 'processed', ntype, 'node_feat.npy')
                out_name = osp.join(args.path, args.dataset_size, 'processed', ntype, 'node_feat.pt')
                nf = torch.from_numpy(np.load(in_name)).to(torch.bfloat16)
                torch.save(nf, out_name)
        elif args.target_dtype == "int8":
            bsz = args.block_size
            for ntype in ntypes:
                in_name = osp.join(args.path, args.dataset_size, 'processed', ntype, 'node_feat.npy')
                out_name = osp.join(args.path, args.dataset_size, 'processed', ntype, 'node_feat_int8.pt')
                scf_name = osp.join(args.path, args.dataset_size, 'processed', ntype, 'node_feat_scf.pt')

                nf = torch.from_numpy(np.load(in_name))
                gnn_utils.quantize_dataset_feat(nf, out_name, scf_name, bsz)

