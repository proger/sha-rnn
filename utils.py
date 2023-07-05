import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def zero_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h * 0
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    #if args.cuda:
    #    data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    if args.cuda:
        data = data.cuda()
        target = target.cuda()
    return data.long(), target.long()

if __name__ == '__main__':
    import data
    corpus = data.Corpus('data/ubertext')

    class args:
        batch_size = 1
        cuda = False
        bptt = 16

    eval_batch_size = min(100, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    print(get_batch(val_data, 0, args))

