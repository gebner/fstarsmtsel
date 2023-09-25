#!/usr/bin/env python3
from typing_extensions import TypedDict, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import json
import random
import wandb, wandb.plot
import pickle

class Decl(TypedDict):
    name: str
    kind: str
    text: str

class Query(TypedDict):
    filename: str
    used_premises: list[str]
    unused_premises: list[str]
    header: str
    query_pre: str
    query: str
    query_fml: str
    query_post: str

class UnsatCoreDataset(TypedDict):
    queries: list[Query]
    decls: dict[str, Decl]

class EmbModel(torch.nn.Module):
    def __init__(self, base_model, pad_token):
        super().__init__()
        self.base_model = base_model
        self.pad_token = pad_token

    def forward(self, input_ids):
        max_len = max((len(toks) for toks in input_ids), default = 0)
        batch = torch.IntTensor([
            toks + [self.pad_token]*(max_len-len(toks))
            for toks in input_ids
        ]).to(self.base_model.device)
        hidden = self.base_model.forward(batch).last_hidden_state
        return F.normalize(torch.stack([
            hidden[i, len(input_ids[i]) - 1, :]
            for i in range(hidden.shape[0])
        ]))

QUERY_EMBEDDING = '<query-embedding>'
PREMISE_EMBEDDING = '<premise-embedding>'

def train(train_ds: UnsatCoreDataset, valid_ds: UnsatCoreDataset, save_dir: Optional[str]):
    model_name = 'EleutherAI/pythia-160m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token_id: tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({
        'additional_special_tokens': [QUERY_EMBEDDING, PREMISE_EMBEDDING],
    })
    base_model = AutoModel.from_pretrained(model_name).to('cuda')
    model = EmbModel(base_model, tokenizer.pad_token_id)

    dots_toks = tokenizer('...').input_ids
    max_len = 2040

    def tokenize_core(txt, tok_id):
        toks = tokenizer(txt).input_ids
        toks.append(tok_id)
        if len(toks) > max_len:
            toks = toks[:max_len/2] + dots_toks + toks[-max_len/2 + len(dots_toks):]
        return toks
    def tokenize_query(txt): return tokenize_core(txt, tokenizer.additional_special_tokens_ids[0])
    def tokenize_premise(txt): return tokenize_core(txt, tokenizer.additional_special_tokens_ids[1])

    def forward_batched(batch, minibatchsize=16):
        return torch.cat([ model.forward(batch[i:i+minibatchsize]) for i in range(0, len(batch), minibatchsize) ])

    lr = 4e-7
    optimizer = torch.optim.Adam(params=list(model.parameters()), lr=lr)

    wandb.init(config = {
        'base_model_name': model_name,
        'learning_rate': lr,
    })

    def validate(ds, prefix, step, ckpt_dir):
        print(f'Validating {prefix} dataset')
        decls = list(ds['decls'].items())
        decl2idx = { d[0]: i for i, d in enumerate(decls) }
        decl_embs = forward_batched([ tokenize_premise(d[1]['text']) for d in decls ], 64)
        query_embs = forward_batched([ tokenize_query(q['query_fml']) for q in ds['queries'] ], 64)
        sims = (query_embs @ decl_embs.T).to('cpu')
        full_recall_dist = []
        full_recall_fract = []
        recall10 = []
        precision10 = []
        for query, sim_dist in zip(ds['queries'], sims):
            sim = torch.argsort(sim_dist, descending=True).tolist()
            used = set(decl2idx[n] for n in query['used_premises'])
            if len(used) == 0: continue
            unused = set(decl2idx[n] for n in query['unused_premises'])
            prems = used | unused
            sim = [ i for i in sim if i in prems ]
            full_recall_fract.append(max((k for k, i in enumerate(sim) if i in used), default=0) / len(prems))
            precision10.append(len([i for i in sim[:10] if i in used]) / 10)
            recall10.append(len([i for i in sim[:10] if i in used]) / len(used))
            full_recall_dist.append(max((sim_dist[i].arccos().item() for i in used), default=0))
        full_recall_fract = torch.Tensor(full_recall_fract)
        full_recall_fract_percentiles = [ [p, torch.quantile(full_recall_fract, p/100).item()] for p in range(101) ]
        full_recall_fract_percentiles = wandb.Table(data=full_recall_fract_percentiles, columns = ["x", "y"])
        full_recall_dist = torch.Tensor(full_recall_dist)
        log = {
            f'{prefix}_mean_full_recall_fract': torch.mean(full_recall_fract).item(),
            f'{prefix}_mean_full_recall_dist': torch.mean(full_recall_dist).item(),
            f'{prefix}_max_full_recall_dist': torch.max(full_recall_dist).item(),
            f'{prefix}_max_full_recall_dist_90percentile': torch.quantile(full_recall_dist, 0.9).item(),
            #'{prefix}_full_recall_fract_50percentile': torch.quantile(full_recall_fract, 0.5).item(),
            f'{prefix}_full_recall_fract_90percentile': torch.quantile(full_recall_fract, 0.9).item(),
            f'{prefix}_full_recall_fract_percentiles': wandb.plot.line(full_recall_fract_percentiles, 'x', 'y',
                title=f'Percentiles of full recall fraction ({prefix} set, step {step})'),
            f'{prefix}_mean_recall_at_10': torch.mean(torch.Tensor(recall10)).item(),
            f'{prefix}_mean_precision_at_10': torch.mean(torch.Tensor(precision10)).item(),
        }
        if ckpt_dir:
            with open(f'{ckpt_dir}/{prefix}.results', 'wb') as f:
                pickle.dump({
                    'decls': [ d[0] for d in decls ],
                    'decl_embs': decl_embs.to('cpu'),
                    'query_embs': query_embs.to('cpu'),
                }, f)
        print(log)
        wandb.log(log, step=step)

    nepochs = 4
    step = 0
    for _ in range(nepochs):
        qs = list(train_ds['queries'])
        random.shuffle(qs)
        for q in qs:
            if (step % 500) == 0:
                ckpt_dir = save_dir and f'{save_dir}/ckpt-{step}'
                if ckpt_dir: base_model.save_pretrained(ckpt_dir)
                with torch.no_grad():
                    validate(valid_ds, 'valid', step, ckpt_dir)
                    validate(train_ds, 'train', step, ckpt_dir)
            step += 1
            optimizer.zero_grad()
            pos_prems = list(q['used_premises'])
            random.shuffle(pos_prems)
            pos_prems = [ tokenize_premise(train_ds['decls'][n]['text']) for n in pos_prems[:20] ]
            k = len(pos_prems)
            if k == 0: continue
            neg_prems = [ tokenize_premise(train_ds['decls'][n]['text'])
                for n in random.choices(q['unused_premises'], k=len(pos_prems)) ]
            query_toks = tokenize_query(q['query_fml'])
            embs = forward_batched(pos_prems + neg_prems + [query_toks])
            # loss = torch.mean(((embs[:-1] @ embs[:-1].T) - torch.eye(embs.shape[0]-1).to(embs.device))**2)
            loss = torch.mean((embs[:k] @ embs[-1] - 1)**2)
            loss += torch.mean((embs[k:2*k] @ embs[-1])**2)
            loss /= 2
            loss.backward()
            optimizer.step()
            wandb.log({
                'train_loss': loss.detach(),
            }, step=step)
            print(f'step {step:5}: loss={loss:.3f} name={q["filename"]}')

if __name__ == '__main__':
    import sys
    train_ds = json.load(open(sys.argv[1]))
    valid_ds = json.load(open(sys.argv[2]))
    train(train_ds, valid_ds, 'outputs')
