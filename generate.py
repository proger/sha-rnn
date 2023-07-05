import argparse
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--words', type=int, default='128',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--file', type=Path, nargs='*')
parser.add_argument('prompt', nargs='*')
args = parser.parse_args()

def model_save(fn):
    with open(fn, 'wb') as f:
        #torch.save([model, criterion, optimizer], f)
        torch.save([model, criterion], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        #model, criterion, optimizer = torch.load(f)
        model, criterion = torch.load(f)
        #model.load_state_dict(m.state_dict(), strict=False)
        #del m

model, criterion = torch.load(args.checkpoint)

model.eval()

if args.cuda:
    model.cuda()
    model.float()
else:
    model.cpu()

import os
import hashlib
#fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
#if os.path.exists(fn):
#    print('Loading cached dataset...')
#    corpus = torch.load(fn)
#else:
#    print('Producing dataset...')
#    corpus = data.Corpus(args.data)
#    torch.save(corpus, fn)
#
#dictionary = corpus.dictionary

from data import Dictionary
dictionary = Dictionary()
for byte in range(256):
    dictionary.add_word(byte.to_bytes(1, byteorder='little'), freq=0)

#del corpus
ntokens = len(dictionary)
hidden = None
mems = None

#print('ready', file=sys.stderr)
#text = sys.stdin.read()

#import youtokentome as yttm
#m = 'data/wpwikitext-103/wt103.yttm'
#bpe = yttm.BPE(model=m)
#text = ' '.join(bpe.encode(text, output_type=yttm.OutputType.SUBWORD))

#if type(text) == str:
#    text = text.encode('utf8')

EOS = ' <eos> '
EOS = '\n'
#text = [str(c) if c != ord('\n') else '<eos>' for c in text]

#text = [w for w in text.replace('\n', EOS).split() if w]

#maxlen = (2 * 1400) - 1
#maxlen = model.num_max_positions
#text = text[-maxlen:]
#orig = ' '.join(w if w != '<eos>' else '\n' for w in text)

#orig = text.encode('utf-8')

#print(text)

#text = [dictionary.word2idx[c] for c in text]

#print(text)

#input = torch.rand(1, 1).mul(ntokens).long()
#print(input.shape)

#input = torch.Tensor(text).view(-1, 1).long()
#input = torch.ByteTensor(text.encode('utf-8'))

def produce_vocab_logits(head_weight, head_bias, hiddens):
    head_res = torch.nn.functional.linear(hiddens, head_weight, bias=head_bias)
    #softmaxed_head_res = torch.nn.functional.log_softmax(head_res, dim=-1)
    #softmaxed_head_res = F.softmax(head_res, dim=-1)
    return head_res

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

context = 'постійно\tp o s t1 i1 j n o\n'
context = 'карликом-наркоторговцем\tk a1 r l y k o m n a r k o t o r h o1 v ts e m\n'
context = 'карасем\tk a r a s e1 m\n'

if args.file:
    prompts = chain(line.split(maxsplit=1)[0]
                      for filename in args.file
                      for line in filename.read_text().split('\n')[:-1])
    prompts = chain(prompts, args.prompt)
else:
    prompts = args.prompt

for input_word in prompts:
    prompt = context + input_word + '\t'
    buffer = np.array(np.frombuffer(prompt.encode('utf-8'), dtype=np.uint8))
    input = torch.from_numpy(buffer).clone().long()[:,None]

    if args.cuda:
        input = input.cuda()

    logits, hidden, mems = model(input[:-1, :], hidden, mems=mems, return_h=False)
    input = input[-1:, :]
    # TODO: We lose a token here as we predict one, update the memory, but don't add it to our generated text

    sys.stdout.buffer.write(input_word.encode('utf-8'))
    sys.stdout.buffer.write(b'\t')

    for i in range(args.words):
        #import ipdb; ipdb.set_trace()
        with torch.no_grad():
            logits, hidden, mems = model(input, hidden, mems=mems, return_h=False)
        # TODO: What if we want to start with no history?
        #magic_mem = []
        #for ma, mb in zip(mems, new_mems):
        #    magic_mem.append(torch.cat([ma, mb], dim=0)[-maxlen:])
        #mems = magic_mem
        output = produce_vocab_logits(model.decoder.weight, model.decoder.bias, logits) / args.temperature
        output = top_k_top_p_filtering(output.view(-1), top_k=1).view(*output.shape)
        #output = top_k_top_p_filtering(output.view(-1), top_p=0.98).view(*output.shape)
        #print('output', output)
        word_weights = F.softmax(output, dim=-1).squeeze()
        #print('word_weights', word_weights)
        #word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, num_samples=1)[0]
        input.data.fill_(word_idx)
        word = dictionary.idx2word[word_idx]

        sys.stdout.buffer.write(word)
        if word == b'\n':
            break
