#!/usr/bin/env python3
from typing_extensions import TypedDict, Literal, Union, Any
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, Trainer, DefaultDataCollator, TrainingArguments
import json
import random
import wandb

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
        return torch.stack([
            hidden[i, len(input_ids[i]) - 1, :]
            for i in range(hidden.shape[0])
        ])
        # hidden = self.base_model.forward(input_ids).last_hidden_state
        # def last_nonzero_id(tokens):
        #     toks: list[int] = tokens.tolist()
        #     if self.pad_token in toks:
        #         return toks.index(self.pad_token) - 1
        #     else:
        #         return -1
        # return torch.stack([
        #     hidden[i, last_nonzero_id(input_ids[i]), :]
        #     for i in range(hidden.shape[0])
        # ])

QUERY_EMBEDDING = '<query-embedding>'
PREMISE_EMBEDDING = '<premise-embedding>'

def train(train_ds: UnsatCoreDataset, valid_ds: UnsatCoreDataset):
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
    if not tokenizer.pad_token_id: tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({
        'additional_special_tokens': [QUERY_EMBEDDING, PREMISE_EMBEDDING],
    })
    base_model = AutoModel.from_pretrained('EleutherAI/pythia-160m', device_map='auto')
    model = EmbModel(base_model, tokenizer.pad_token_id)

    def tokenize_core(txt, tok_id):
        toks = tokenizer(txt, max_length=200, truncation=True).input_ids
        toks.append(tok_id)
        return toks
    def tokenize_query(txt): return tokenize_core(txt, tokenizer.additional_special_tokens_ids[0])
    def tokenize_premise(txt): return tokenize_core(txt, tokenizer.additional_special_tokens_ids[1])

    def forward_batched(batch, minibatchsize=8):
        return torch.cat([ model.forward(batch[i:i+minibatchsize]) for i in range(0, len(batch), minibatchsize) ])

    lr = 1e-5
    optimizer = torch.optim.Adam(params=list(model.parameters()), lr=lr)

    wandb.init()

    def validate():
        decls = list(valid_ds['decls'].items())
        random.shuffle(decls) # TODO: filter decls for ones actually occurring in file
        decl2idx = { d[0]: i for i, d in enumerate(decls) }
        # print('declaration embeddings')
        decl_embs = forward_batched([ tokenize_premise(d[1]['text']) for d in decls ], 64)
        # print('query embeddings')
        query_embs = forward_batched([ tokenize_query(q['query_fml']) for q in valid_ds['queries'] ], 64)
        # print('validation')
        sims = (F.normalize(query_embs) @ F.normalize(decl_embs).T).to('cpu')
        full_recall_fract = []
        for query, sim in zip(valid_ds['queries'], sims):
            sim = torch.argsort(sim, descending=True).tolist()
            # print(query['filename'])
            # print(query['used_premises'])
            # print([ decls[i][0] for i in sim[:10] ])
            used = [decl2idx[n] for n in query['used_premises']]
            if len(used) == 0: continue
            full_recall_fract.append(max(sim.index(u) for u in used) / len(decls))
        log = {
            'valid_mean_full_recall_fract': torch.mean(torch.Tensor(full_recall_fract)).item(),
        }
        print(log)
        wandb.log(log)

    nepochs = 3
    step = 0
    for _ in range(nepochs):
        qs = list(train_ds['queries'])
        random.shuffle(qs)
        for q in qs:
            if (step % 100) == 0:
                with torch.no_grad():
                    validate()
            step += 1
            optimizer.zero_grad()
            pos_prems = q['used_premises']
            random.shuffle(pos_prems)
            pos_prems = [ tokenize_premise(train_ds['decls'][n]['text']) for n in pos_prems[:15] ]
            k = len(pos_prems)
            if k == 0: continue
            neg_prems = [ tokenize_premise(train_ds['decls'][n]['text'])
                for n in random.choices(q['unused_premises'], k=len(pos_prems)) ]
            query_toks = tokenize_query(q['query_fml'])
            embs = forward_batched(pos_prems + neg_prems + [query_toks])
            loss = torch.mean(torch.cat([
                torch.stack([
                    F.cosine_embedding_loss(embs[-1].reshape((1,-1)), embs[i].reshape((1,-1)), torch.Tensor([1]).to(embs.device)),
                    F.cosine_embedding_loss(embs[-1].reshape((1,-1)), embs[k+i].reshape((1,-1)), torch.Tensor([-1]).to(embs.device)),
                ])
                for i in range(k)
            ]))
            loss.backward()
            optimizer.step()
            wandb.log({
                'train_loss': loss.detach(),
            }, step=step)
            print(f'step {step:5}: loss={loss:.3f} name={q["filename"]}')

if __name__ == '__main__':
    train_ds = json.load(open('ulib.json'))
    valid_ds = json.load(open('merkle-tree.json'))
    train(train_ds, valid_ds)
