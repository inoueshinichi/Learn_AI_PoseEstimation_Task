"""生データのディスクキャッシュ
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
# print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

from .log_conf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


import gzip

from diskcache import FanoutCache, Disk
from diskcache.core import MODE_BINARY

# カサンドラ: 分散データベース
from cassandra.cqltypes import BytesType

from io import BytesIO

from .type_hint import *


class GzipDisk(Disk):
    def __init__(self,
                 compress_level: int = 1,
                 **kwargs,
                 ):
        super().__init__(kwargs)
        
        self.compress_level: int = compress_level
        self.chunk_size: int = 2**30
    
    # override
    def store(self,
              value: Any,
              read: bool,
              key: Optional[Any] = None,
              ):
        if type(value) is BytesType:
            # バイナリデータの場合, 圧縮
            if read:
                value = value.read()
                read = False
            
            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode='wb',
                                    compresslevel=self.compress_level,
                                    fileobj=str_io,
                                    )
            
            for offset in range(0, len(value), self.chunk_size):
                gz_file.write(value[offset : offset + self.chunk_size])
            gz_file.close()

            value = str_io.getvalue()

        return super().store(value=value, read=read, key=key)

    # override
    def fetch(self,
              mode: int,
              filename: str,
              value: Any,
              read: bool,
              ):
        value = super().fetch(mode,
                              filename,
                              value,
                              read,
                              )
        if mode == MODE_BINARY:
            # バイナリデータの場合, 解凍
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode='rb',
                                    fileobj=str_io,
                                    )
            read_csio = BytesIO() # Cassandra

            while True:
                uncompressed_data = gz_file.read(self.chunk_size)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break
            
            value = read_csio.getvalue()
        
        return value
    

def make_disk_cache(cache_dir: str,
                   scope: str,
                   version: str,
                   ):
    caching_dir = cache_dir + f'/cache-{scope}/{version}'
    return FanoutCache(
        directory=caching_dir,
        disk=GzipDisk,
        shards=64, # 64 db partition
        timeout=1, # 1sec
        size_limit=3e11, # max total db size
    )


