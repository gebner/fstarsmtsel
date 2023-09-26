#!/usr/bin/env python3
import json
import re
import itertools, sys

comment = re.compile(r';.*$', re.M)
query_fml = re.compile(r'^; Encoding query formula : (.*\n(; .*\n)*)', re.M)
named_pat = re.compile(r':named ([^ )]*)')

chunk_pats = {
    'start': re.compile(r'^; <start (\S+) (\S*)>\n', re.I|re.M),
    'end': re.compile(r'^; </end (\S+) (\S*)>\n', re.I|re.M),
    'push': re.compile(r'^\(push\)\n', re.M),
    'pop': re.compile(r'^\(pop\)\n', re.M),
    'query': re.compile(r'^; Starting query at (\S+)\n(.*?); UNSAT CORE GENERATED: (.*?)\n', re.S|re.M),
    'precedes': re.compile(r'^\(declare-fun Prims\.precedes.*:qid __prelude_valid_intro\)\)\)', re.S | re.M),
}

def chunk_file(src: str):
    i = 0
    chunks = []
    cache = { k: r.search(src) for k, r in chunk_pats.items() }
    while i < len(src):
        k, m = None, None
        for k2, r2 in chunk_pats.items():
            m2 = cache.get(k2)
            if m2 and m2.start() < i:
                m2 = r2.search(src, i)
                cache[k2] = m2
            if not m2: continue
            if not m or m2.start() < m.start():
                k, m = k2, m2
        if m:
            chunks.append(('text', src[i:m.start()]))
            chunks.append((k, m))
            i = m.end()
        else:
            chunks.append(('text', src[i:]))
            break
    return chunks

def strip_comments(smt: str):
    return comment.sub('', smt)

def parse_smt2(src: str):
    header = None
    chunks = []
    scope_chunks = []
    cur_chunk = []
    enc_stack = []
    push_stack = []
    queries = []

    for c in chunk_file(src):
        if header is None:
            assert c[0] == 'text'
            header = c[1]
            continue
        
        if c[0] == 'start':
            enc_stack.append(c[1].group(2))
            cur_chunk.append(c[1].group())
            continue
        
        if c[0] == 'end':
            if len(enc_stack) > 1:
                closing, opening = c[1].group(2), enc_stack.pop()
                if closing != opening:
                    raise Exception(f'mismatch <start> </end>: {opening} vs {closing}')
            else:
                assert len(enc_stack) == 1
                text = ''.join(cur_chunk).strip()
                pat_names = named_pat.findall(text)
                scope_chunks.append(len(chunks))
                chunks.append({
                    'name': enc_stack[0],
                    'text': text,
                    'pat_names': pat_names,
                })
                enc_stack, cur_chunk = [], []
            continue

        if c[0] == 'push':
            assert len(enc_stack) == 0
            push_stack.append(list(scope_chunks))
            continue

        if c[0] == 'pop':
            assert len(enc_stack) == 0
            assert len(push_stack) > 0
            scope_chunks = push_stack.pop()
            continue
    
        if c[0] == 'text' and len(enc_stack) > 0:
            cur_chunk.append(c[1])
            continue
    
        if c[0] == 'text' and strip_comments(c[1]).strip() == '':
            continue

        if c[0] == 'query':
            assert len(enc_stack) == 0
            text = c[1].group(2)
            fml = query_fml.search(text)
            fml = fml[1].replace('\n; ', '\n').strip()
            queries.append({
                'premises': list(scope_chunks),
                'loc': c[1].group(1),
                'text': text,
                'fml': fml,
                'core': c[1].group(3).split(', '),
            })
            continue

        if c[0] == 'precedes':
            assert len(enc_stack) == 0
            text = c[1].group()
            pat_names = named_pat.findall(text)
            scope_chunks.append(len(chunks))
            chunks.append({
                'name': 'precedes',
                'text': text,
                'pat_names': pat_names,
            })
            continue

        raise Exception(f'unexpected chunk {c}')
    
    # assert len(push_stack) == 0
    assert len(enc_stack) == 0

    return {
        'header': header,
        'premises': chunks,
        'queries': queries,
    }

def parse_smt2_files(fns):
    decls = {}
    out = []

    def intern_decl(c):
        nonlocal decls
        n = c['name']
        for i in itertools.count(0):
            n2 = n if i == 0 and n != '' else f'{n}[{i}]'
            if n2 not in decls:
                decls[n2] = c
                return n2
            if decls[n2]['text'] == c['text']:
                return n2

    for fn in fns:
        try:
            queries = parse_smt2(open(fn).read())
        except Exception as e:
            sys.stderr.write(str(Exception(fn, e)) + '\n')
            continue

        prems = [ {
            'pat_names': prem['pat_names'],
            'name': intern_decl(prem),
        } for prem in queries['premises'] ]

        for query in queries['queries']:
            core: list[str] = query['core']
            used_premises = []
            unused_premises = []

            for c in query['premises']:
                c = prems[c]
                used = any(pn in core for pn in c['pat_names'])
                (used_premises if used else unused_premises).append(c['name'])

            out.append({
                'filename': fn,
                'used_premises': used_premises,
                'unused_premises': unused_premises,
                'header': queries['header'],
                'loc': query['loc'],
                'query_full': query['text'],
                'query_fml': query['fml'],
            })

    return {
        'decls': decls,
        'queries': out,
    }

if __name__ == '__main__':
    import sys
    import json
    json.dump(parse_smt2_files(sys.argv[1:]), sys.stdout)
