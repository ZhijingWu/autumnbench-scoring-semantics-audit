"""Audit recorded native executions without rebuilding the native interpreter."""
import gzip,json,pathlib,shutil,subprocess,sys,tempfile
root=pathlib.Path(__file__).resolve().parent
from verify_manifest import verify
verify(root)
(root/'work').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=root/'work',prefix='autumn-replay-') as tmp:
 d=pathlib.Path(tmp)/'study';shutil.copytree(root,d,ignore=shutil.ignore_patterns('.git','work','__pycache__'))
 for p in d.rglob('*.gz'):
  if p.name.endswith(('.json.gz','.jsonl.gz')):p.with_suffix('').write_bytes(gzip.decompress(p.read_bytes()))
 expected=json.loads((d/'ANALYSIS_SUMMARY.json').read_text())
 for name in ['ANALYSIS_SUMMARY.json','SCENARIO_RESULTS.json']:(d/name).unlink()
 subprocess.run([sys.executable,'-B',str(d/'scripts/analyze_results.py')],check=True,stdout=subprocess.DEVNULL)
 actual=json.loads((d/'ANALYSIS_SUMMARY.json').read_text());assert actual==expected
 assert actual['factorial_scenarios']==136 and actual['native_call_total']==21513
 print('PASS: 136 factorial scenarios, three controls, eight paired replays and 21,513 native calls reaudited; no native/world/model execution.')
