import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
#import torchmetrics.functional as MF
import time, tqdm, numpy as np
from models import *
from dataset import IGBHeteroDGLDataset
import tpp_pytorch_extension as ppx
from dgl.dataloading import NeighborSampler
import os, psutil
import collections

torch.manual_seed(0)
dgl.seed(0)
import warnings
warnings.filterwarnings("ignore")

def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

def evaluate(model, dataloader):
    predictions = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']
            labels.append(blocks[-1].dstdata['label']['paper'].cpu().numpy())
            predictions.append(model(blocks, inputs).argmax(1).cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        acc = sklearn.metrics.accuracy_score(labels, predictions)
        return acc

def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    if args.opt_mlp:
        batch_inputs = gnn_utils.gather_features(nfeat, input_nodes)
    else:
        batch_inputs = nfeat[input_nodes]

    batch_labels = labels[seeds]

    return batch_inputs, batch_labels

def load_subtensor_dict(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs={}
    ntypes = nfeat.keys()
    if args.opt_mlp:
        for ntype in ntypes:
            if args.use_bf16:  nfeat[ntype] = nfeat[ntype].to(torch.bfloat16)
            batch_inputs[ntype] = gnn_utils.gather_features(nfeat[ntype], input_nodes[ntype])
    else:
        batch_inputs = nfeat[input_nodes]

    batch_labels = labels[seeds]

    return batch_inputs, batch_labels

def shuffle(ten):
    idx = torch.randperm(ten.shape[0])
    ten = ten[idx]
    return ten

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def track_acc(g, category, args, device):

    fanouts = [int(fanout) for fanout in args.fan_out.split(",")]
    sampler = NeighborSampler(fanouts)

    train_nid = torch.nonzero(g.nodes[category].data['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.nodes[category].data['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.nodes[category].data['test_mask'], as_tuple=True)[0]

    if not args.opt_mlp:
        train_dataloader = dgl.dataloading.DataLoader(
            g, {category: train_nid}, sampler,
            batch_size=args.batch_size,
            shuffle=True, drop_last=False,
            num_workers=args.num_workers)

    et_sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    val_dataloader = dgl.dataloading.DataLoader(
        g, {category: val_nid}, et_sampler,
        batch_size=args.val_batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, {category: test_nid}, et_sampler,
        batch_size=args.val_batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)

    nfeat = g.ndata['feat']
    labels = g.ndata['label'][category]
    in_feats = g.ndata['feat'][category].shape[1]

    if args.model_type == 'rgcn':
        model = RGCN(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rsage':
        model = RSAGE(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rgat':
        with opt_impl(args.opt_mlp, args.use_bf16):
            model = RGAT(g.etypes, in_feats, args.hidden_channels,
                args.num_classes, args.num_layers, args.num_heads, F.leaky_relu).to(device)
    print(model)
    if args.opt_mlp:
        block(model)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    loss_fcn = nn.CrossEntropyLoss().to(device)
    if args.opt_mlp:
        no_decay = ["bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = ppx.optim.Adam(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
    else:
        optimizer = optim.Adam(model.parameters(),
            lr=args.learning_rate)
    sched = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.25)

    best_accuracy = 0
    training_start = time.time()
    for epoch in range(args.epochs):
        batch_fwd_time = AverageMeter()
        batch_bwd_time = AverageMeter()
        data_time = AverageMeter()
        gather_time = AverageMeter()

        model.train()
        total_loss = 0
        train_acc = 0
        idx = 0

        epoch_start = time.time()
        train_nid_ = shuffle(train_nid)

        end = time.time()

        #ppx.reset_debug_timers()
        if args.opt_mlp:
            for step_ in range(0, train_nid.shape[0], args.batch_size):
                if args.opt_mlp and epoch == 0 and step_ == 0:
                    cores = int(os.environ["OMP_NUM_THREADS"])
                    gnn_utils.affinitize_cores(cores, 0)
                step = int(step_ / args.batch_size)
                if step_ + args.batch_size < train_nid.shape[0]:
                    seeds = train_nid_[step_: step_ + args.batch_size]
                else:
                    seeds = train_nid_[step_:]
                seeds_dict = {category:seeds}
                idx += 1
                #blocks = sampler.sample_blocks_syncord(seed_dict, category)
                seeds_dict, output_nodes, blocks = sampler.sample_blocks(g, seeds_dict)
                input_nodes = blocks[0].srcdata[dgl.NID]
                t0 = time.time()
                data_time.update(t0 - end)

                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor_dict(
                    nfeat, labels, seeds, input_nodes
                )
                t1 = time.time()
                gather_time.update(t1 - t0)

                t2 = time.time() 

                batch_pred = model(blocks, batch_inputs)
                
                t3 = time.time()
                batch_fwd_time.update(t3 - t2)

                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                
                t4 = time.time()
                loss.backward()
                t5 = time.time()
                batch_bwd_time.update(t5 - t4)

                optimizer.step()
                end = time.time()

                total_loss += loss.item()
                train_acc += sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(),
                    batch_pred.argmax(1).detach().cpu().numpy())*100
                if step % args.log_every == 0:
                    print("Epoch {:05d} | Step {:05d} | Loss {:.2f} | "
                          "DL (s) {data_time.val:.3f} ({data_time.avg:.3f}) | "
                          "GT (s) {gather_time.val:.3f} ({gather_time.avg:.3f}) | "
                          "FWD (s) {batch_fwd_time.val:.3f} ({batch_fwd_time.avg:.3f}) | "
                          "BWD (s) {batch_bwd_time.val:.3f} ({batch_bwd_time.avg:.3f}) | ".format(
                               epoch,
                               step,
                               loss.item(),
                               data_time=data_time,
                               gather_time=gather_time,
                               batch_fwd_time=batch_fwd_time,
                               batch_bwd_time=batch_bwd_time,
                            )
                    )
        else:
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                idx += 1
                blocks = [block.to(device) for block in blocks]
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label']['paper']
                y_hat = model(blocks, x)
                loss = loss_fcn(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                train_acc += sklearn.metrics.accuracy_score(y.cpu().numpy(),
                    y_hat.argmax(1).detach().cpu().numpy())*100

        #ppx.print_debug_timers(0)
        train_acc /= idx
        epoch_end = time.time()
        print("Epoch {:05d} time {:.2f}".format(epoch, epoch_end - epoch_start))

        if epoch%args.eval_every == 0:
            model.eval()
            val_acc = evaluate(model, val_dataloader)
            if best_accuracy < val_acc:
                best_accuracy = val_acc
                if args.model_save:
                    torch.save(model.state_dict(), args.modelpath)

            tqdm.tqdm.write(
                "Epoch {:03d} | Loss {:.4f} | Train Acc {:.2f} | Val Acc {:.2f} | Time {} ".format(
                    epoch,
                    total_loss,
                    train_acc,
                    val_acc,
                    str(datetime.timedelta(seconds = int(time.time() - epoch_start))),
                )
            )
        sched.step()

    model.eval()
    test_acc = evaluate(model, test_dataloader).item()*100
    print("Test Acc {:.2f}%".format(test_acc))
    print("Total time taken " + str(datetime.timedelta(seconds = int(time.time() - training_start))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/root/gnndataset',
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')

    # Model
    parser.add_argument('--model_type', type=str, default='rgat',
                        choices=['rgat', 'rsage', 'rgcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters
    parser.add_argument('--fan_out', type=str, default='15,10')
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument('--val_batch_size', type=int, default=10240)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument( "--adam_epsilon", default=1e-8, type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument( "--opt_mlp", action="store_true",
        help="Whether to use optimized MLP impl when available",
    )
    parser.add_argument( "--use_bf16", action="store_true",
        help="Whether to use BF16 datatype when available",
    )
    parser.add_argument('--all_in_edges', type=bool, default=True,
         help="Set to false to use default relation. Set this option to True " 
         "to use all the relation types in the dataset since DGL samplers require directed in edges.")
    args = parser.parse_args()

    device = f'cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'
    use_label_2K = int(args.num_classes) == 2983
    dataset = IGBHeteroDGLDataset(args.path, args.dataset_size, args.in_memory, use_label_2K, args.use_bf16)
    g = dataset[0]
    category = g.predict

    track_acc(g, category, args, device)
