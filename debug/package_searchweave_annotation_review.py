"""Package annotation export for review without changing its original file.

Fill absent render URLs from the node's original image URL, retain provenance
warnings, and include the cached local images as an offline backup.
"""
import json
import shutil
from pathlib import Path

P = Path('data/searchweave_v2_candidates_temp')


def main():
    source = P / 'annotation_ready.jsonl'
    output = P / 'annotation_ready_for_review.jsonl'
    raw = [json.loads(l) for l in source.read_text().splitlines() if l.strip()]
    audit, copied = [], {}
    before_qa = [(r['question_id'], r['question'], r['answer']) for r in raw]
    images_dir = P / 'local_images'
    images_dir.mkdir(exist_ok=True)

    def fill(value, path, changes):
        if isinstance(value, dict):
            if 'image_url' in value and not value['image_url'] and value.get('original_url', '').startswith(('http://', 'https://')):
                value['image_url'] = value['original_url']
                changes.append({'field': path + '.image_url', 'url': value['image_url']})
            for k, v in value.items():
                fill(v, path + '.' + k, changes)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                fill(v, path + f'[{i}]', changes)

    fixed_nodes = 0
    for r in raw:
        changes = []
        fixed_nodes += sum(n['node_type'] == 'image' and not n.get('image_url') for n in r['search_chain']['nodes'])
        fill(r, '', changes)
        backups = []
        for w in r.get('warnings', []):
            if not w.startswith('local_image_upload_not_configured:'):
                continue
            cached = Path(w.split(':', 1)[1])
            assert cached.is_file(), cached
            target = images_dir / cached.name
            if str(cached) not in copied:
                shutil.copy2(cached, target)
                assert target.read_bytes() == cached.read_bytes()
                copied[str(cached)] = str(target)
            backups.append({'original_cache_file': str(cached), 'packaged_file': str(target)})
        if changes:
            r['warnings'] = [w for w in r.get('warnings', []) if not w.startswith('image_url_missing:')]
            r['warnings'].append('image_url_original_source_fallback')
            audit.append({'question_id': r['question_id'], 'path_id': r['path_id'],
                          'changed_fields': changes, 'local_cache_backups': backups})
    assert len(raw) == 1669 and len({r['question_id'] for r in raw}) == 1669
    assert [(r['question_id'], r['question'], r['answer']) for r in raw] == before_qa
    assert all(n.get('image_url') for r in raw for n in r['search_chain']['nodes'] if n['node_type'] == 'image')
    tmp = output.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in raw))
    tmp.replace(output)
    (P / 'image_url_fallback_audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in audit))
    summary = {'records': 1669, 'affected_samples': len(audit), 'fixed_image_nodes': fixed_nodes,
               'local_image_files_copied': len(copied), 'review_file': str(output),
               'original_export': str(source), 'original_export_changed': False,
               'all_image_nodes_have_render_url': True,
               'external_image_urls_live_checked': False,
               'note': 'Fallback URLs are the original image URLs already stored in the export. Source-evidence and local-upload provenance warnings remain; no record was discarded.'}
    (P / 'review_package_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (P / 'README.md').open('a') as f:
        f.write('\nCompleted annotation export: annotation_ready.jsonl (1669 rows; all translations complete).\n'
                'For labeling, use annotation_ready_for_review.jsonl: missing primary image URLs are filled from stored original image URLs.\n'
                'Cached images for these cases are included in local_images; correspondence is recorded in image_url_fallback_audit.jsonl.\n'
                'All records retain incomplete status because of unresolved source graph evidence references. These are not translation failures.\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
