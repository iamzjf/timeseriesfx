"""Download the official CUDA wheel in verified HTTP ranges; no TLS bypass."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import time
import urllib.request

URL = 'https://download.pytorch.org/whl/cu121/torch-2.4.1%2Bcu121-cp38-cp38-win_amd64.whl'
SHA256 = 'a48b991cd861266523cbed4705f89bef09669d5d2bbfa2524486156f74a222a8'
DEST = Path(r'E:\conda_envs\wheelhouse_tsfx\torch-2.4.1+cu121-cp38-cp38-win_amd64.whl')
SIZE = 2444894201
CHUNK = 16 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size == SIZE and sha256(DEST) == SHA256:
        print('Official CUDA wheel already verified.', flush=True)
        return
    # Parts make interrupted downloads resumable; an existing sequential partial is retained.
    parts = DEST.parent / 'torch_parts'
    parts.mkdir(exist_ok=True)
    def fetch(i):
        start = i * CHUNK
        end = min(SIZE, start + CHUNK) - 1
        part = parts / ('%04d.part' % i)
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(4):
            try:
                req = urllib.request.Request(URL, headers={'Range': 'bytes=%d-%d' % (start, end)})
                with urllib.request.urlopen(req, timeout=45) as resp, part.open('wb') as f:
                    if resp.status != 206 or resp.headers.get('Content-Range') != 'bytes %d-%d/%d' % (start, end, SIZE):
                        raise RuntimeError('Server did not honor exact range')
                    while True:
                        buf = resp.read(256 * 1024)
                        if not buf:
                            break
                        f.write(buf)
                if part.stat().st_size != end - start + 1:
                    raise RuntimeError('Incomplete download range')
                return part
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    total = (SIZE + CHUNK - 1) // CHUNK
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch, i) for i in range(total)]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if count % 8 == 0 or count == total:
                print('downloaded %d/%d chunks' % (count, total), flush=True)
    tmp = DEST.with_suffix('.verified-download')
    with tmp.open('wb') as out:
        for i in range(total):
            with (parts / ('%04d.part' % i)).open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    out.write(block)
    actual = sha256(tmp)
    if actual != SHA256:
        raise RuntimeError('SHA256 mismatch: ' + actual)
    os.replace(str(tmp), str(DEST))
    print('VERIFIED ' + actual, flush=True)


if __name__ == '__main__':
    main()
