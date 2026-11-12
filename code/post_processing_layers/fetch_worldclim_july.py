"""Download only July from the official 4.38 GB historical archive.

HTTP ranges avoid downloading the other months. ZIP CRC is checked by zipfile.
Usage: python fetch_worldclim_july.py OUTPUT_DIRECTORY
"""
import io
import json
import sys
import zipfile
from pathlib import Path
import requests
from valuation_accounting import sha256

URL = 'https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_tmax.zip'


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.session = requests.Session()
        h=self.session.head(URL,timeout=30); h.raise_for_status()
        self.length=int(h.headers['Content-Length']); self.pos=0
        self.etag=h.headers.get('ETag'); self.modified=h.headers.get('Last-Modified')
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset
        return self.pos
    def read(self,size=-1):
        if size<0: size=self.length-self.pos
        if size==0 or self.pos>=self.length: return b''
        end=min(self.pos+size,self.length)-1
        r=self.session.get(URL,headers={'Range':f'bytes={self.pos}-{end}',
            'If-Match':self.etag},timeout=120)
        r.raise_for_status()
        if r.status_code!=206: raise ValueError('Server did not honor range request')
        data=r.content
        assert len(data)==end-self.pos+1
        self.pos+=len(data);return data


if __name__=='__main__':
    dest=Path(sys.argv[1]); dest.mkdir(parents=True,exist_ok=True)
    remote=RemoteZip()
    with zipfile.ZipFile(remote) as z:
        member=z.getinfo('wc2.1_30s_tmax_07.tif')
        print('July uncompressed/transfer bytes:',member.file_size,member.compress_size,flush=True)
        path=dest/member.filename
        if path.exists(): raise FileExistsError(path)
        with z.open(member) as src,path.open('xb') as out:
            while chunk:=src.read(8*1024*1024): out.write(chunk)
        record=dict(url=URL,archive_bytes=remote.length,etag=remote.etag,
            last_modified=remote.modified,member=member.filename,zip_crc32=member.CRC,
            sha256=sha256(path),size_bytes=path.stat().st_size,
            period='1970–2000',variable='July monthly maximum temperature',units='degrees Celsius',
            source_page='https://www.worldclim.org/data/worldclim21.html')
        (dest/'worldclim_july_download_manifest.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record,indent=2),flush=True)
