#!/usr/bin/env python3
import json
import re

begin_sec = re.compile(r'^; <start (\S+) (.*)>$', re.I | re.M)
end_sec = re.compile(r'^; </(?:\S*) (\S+) (.*?)>$', re.I | re.M)
comment = re.compile(r';.*$', re.M)
query_begin = re.compile(r'; Starting query.*$', re.M)
query_fml = re.compile(r'^; Encoding query formula : (.*\n(; .*\n)*)', re.M)
query_actual_begin = re.compile(r'^;;;*query$', re.M)
query_end = re.compile(r'^\(set-option', re.M)
unsat_core = re.compile(r'^; UNSAT CORE GENERATED: (.*)$', re.M)
named_pat = re.compile(r':named ([^ )]*)')

def strip_comments(smt: str):
    return comment.sub('', smt)

def parse_smt2(src: str):
    i = 0

    def skip_until_after(regex: re.Pattern):
        nonlocal i
        match = regex.search(src, i)
        if not match: return None
        skipped = src[i:match.start()]
        i = match.end()
        return skipped, match

    buf = ''
    def skip_until_after2(regex: re.Pattern):
        skipped = skip_until_after(regex)
        if not skipped: return
        nonlocal buf
        buf += skipped[0]
        return skipped[1]

    chunks = []
    header = ''
    while True:
        skipped = skip_until_after2(begin_sec)
        if header == '': header, buf = buf, ''
        if not skipped: break
        chunk = skip_until_after(end_sec)
        if not chunk: raise Exception(f'cannot find end')

        chunks.append({
            'name': chunk[1][2],
            'kind': chunk[1][1],
            'text': chunk[0].strip(),
        })

    if not skip_until_after2(query_begin): raise Exception('cannot find query')
    buf = comment.sub(buf, '').strip()
    if buf != '': raise Exception('unexpected content {buf}')

    fml = skip_until_after2(query_fml)
    if not fml: raise Exception('cannot find query fml')
    fml = fml[1].replace('\n; ', '\n').strip()

    if not skip_until_after2(query_actual_begin): raise Exception('cannot find query actual_begin')
    query_pre, buf = buf, ''
    end_match = skip_until_after2(query_end)
    if not end_match: raise Exception('cannot find query end')
    query, buf = buf, end_match[0]

    core = skip_until_after2(unsat_core)
    if not core: raise Exception('cannot find unsat core')
    query_post, buf = buf, ''
    core = core[1].split(', ')

    return {
        'header': header,
        'query_pre': query_pre,
        'query': strip_comments(query),
        'query_fml': fml,
        'query_post': query_post,
        'chunks': chunks,
        'unsat_core': core,
    }

import itertools, sys

def parse_smt2_files(fns):
    decls = {}
    queries = []

    def intern_decl(c):
        nonlocal decls
        n = c['name']
        for i in itertools.count(0):
            n2 = n if i == 0 else f'{n}[{i}]'
            if n2 not in decls:
                decls[n2] = c
                return n2
            if decls[n2]['text'] == c['text']:
                return n2

    for fn in fns:
        try:
            query = parse_smt2(open(fn).read())
        except Exception as e:
            sys.stderr.write(str(Exception(fn, e)) + '\n')
            continue

        core: list[str] = query['unsat_core']
        used_premises = []
        unused_premises = []

        for c in query['chunks']:
            n = intern_decl(c)
            pat_names = named_pat.findall(c['text'])
            used = any(pn in core for pn in pat_names)
            (used_premises if used else unused_premises).append(n)

        queries.append({
            'filename': fn,
            'used_premises': used_premises,
            'unused_premises': unused_premises,
            'header': query['header'],
            'query_pre': query['query_pre'],
            'query': query['query'],
            'query_fml': query['query_fml'],
            'query_post': query['query_post'],
        })

    return {
        'decls': decls,
        'queries': queries,
    }

if __name__ == '__main__':
    import sys
    import json
    json.dump(parse_smt2_files(sys.argv[1:]), sys.stdout)
