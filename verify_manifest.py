"""Verify published bytes without changing source evidence."""
import hashlib,json,pathlib
def verify(root=None):
 root=pathlib.Path(root or pathlib.Path(__file__).resolve().parent)
 manifest=json.loads((root/'PUBLIC_MANIFEST.json').read_text(encoding='utf-8'))
 for name,digest in manifest['files'].items():
  p=root/name
  assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
 return len(manifest['files'])
if __name__=='__main__':print('PASS:',verify(),'published file hashes')
