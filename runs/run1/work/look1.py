import json, sys
d = sys.argv[1]
for m in ['base', 'edited']:
    rows = [json.loads(l) for l in open(f'{d}/rows_{m}.jsonl')]
    print(m, list(rows[0].keys()))
    for r in rows:
        print(r['id'], r['tokens'], r['closed'], r['cut'], round(r.get('repetition', r.get('rep', 0)) if not isinstance(r.get('repetition'), type(None)) else 0, 2), r['correct'], repr(r['last_800'][-160:]) if 'last_800' in r else '')
