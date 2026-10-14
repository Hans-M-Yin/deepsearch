"""Merge reasoning-only GPT turns into the following reasoning/tool-call turn."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

from synthesis.post_process.fix.storage import write_json


def reasoning_body(value):
    # Accept only an outer thinking block with optional duplicated opening tags.
    match = re.fullmatch(r'\s*(?:<thinking>\s*)+(.*?)\s*</thinking>\s*', value, re.S)
    if not match or re.search(r'</?thinking>|<tool_call>|<answer>|<image>', match[1]):
        raise ValueError('Unexpected reasoning-only structure')
    return match[1]


def normalize(records):
    output = deepcopy(records)
    affected, merges, duplicate_tags = [], 0, 0
    for index, (original, record) in enumerate(zip(records, output)):
        turns = []
        for message in record['conversations']:
            if turns and turns[-1]['from'] == message['from'] == 'gpt':
                previous = turns.pop()
                if set(previous) != {'from', 'value'} or set(message) != {'from', 'value'}:
                    raise ValueError('Unexpected assistant message metadata')
                match = re.fullmatch(r'(.*?)\s*(<tool_call>.*?</tool_call>)\s*', message['value'], re.S)
                if not match or message['value'].count('<tool_call>') != 1:
                    raise ValueError(f'Unexpected tool-call structure at record {index}')
                a, b = reasoning_body(previous['value']), reasoning_body(match[1])
                duplicate_tags += previous['value'].count('<thinking>') + match[1].count('<thinking>') - 2
                message['value'] = '<thinking>\n' + a + '\n\n' + b + '\n</thinking>\n' + match[2]
                merges += 1
                if not affected or affected[-1] != index:
                    affected.append(index)
            turns.append(message)
        record['conversations'] = turns
        if len(turns) % 2 or any(m['from'] not in (('human', 'observation') if j % 2 == 0 else ('gpt', 'function_call')) for j, m in enumerate(turns)):
            raise ValueError(f'Invalid role alternation at record {index}')
        # Preserve every tool observation and every non-whitespace body character;
        # only thinking wrappers and assistant message boundaries may change.
        for role in ['human', 'observation', 'function_call']:
            assert [m for m in turns if m['from'] == role] == [m for m in original['conversations'] if m['from'] == role]
        def body(messages):
            text = ''.join(m['value'] for m in messages if m['from'] == 'gpt')
            return re.sub(r'\s+', '', re.sub(r'</?thinking>', '', text))
        assert body(turns) == body(original['conversations'])
        def calls(messages):
            return [c for m in messages for c in re.findall(r'<tool_call>.*?</tool_call>', m['value'], re.S)]
        assert calls(turns) == calls(original['conversations'])
        assert record['images'] == original['images']
        assert sum(m['value'].count('<image>') for m in turns) == len(record['images'])
    return output, {'records': len(records), 'affected_records': len(affected),
                    'merged_pairs': merges, 'removed_duplicate_thinking_tags': duplicate_tags,
                    'affected_indices': affected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--backup', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if len({p.resolve() for p in [args.input, args.backup, args.report]}) != 3:
        raise ValueError('Input, backup and report must be distinct')
    records = json.loads(args.input.read_text())
    output, report = normalize(records)
    args.backup.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open('rb') as source, args.backup.open('xb') as backup:
        shutil.copyfileobj(source, backup)
    write_json(args.input, output)
    write_json(args.report, report)
    print(json.dumps({k:v for k,v in report.items() if k != 'affected_indices'}))


if __name__ == '__main__':
    main()
