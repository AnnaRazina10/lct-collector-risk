"""Read selected row identities from the frozen v2 cache with bounded memory."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parquet_rows(path, indexes, columns=None, batch_size=65536):
    requested=np.asarray(indexes,dtype=np.int64)
    if len(np.unique(requested))!=len(requested) or (requested<0).any():
        raise ValueError('Expected unique nonnegative positional row indexes')
    wanted=np.sort(requested);parts=[];offset=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,columns=columns):
        stop=offset+len(batch)
        lo=np.searchsorted(wanted,offset);hi=np.searchsorted(wanted,stop)
        if hi>lo:
            positions=wanted[lo:hi]
            part=batch.take(positions-offset).to_pandas()
            part['_source_row']=positions
            parts.append(part)
        offset=stop
    if not parts:raise ValueError('No requested rows found')
    result=pd.concat(parts,ignore_index=True).set_index('_source_row')
    if len(result)!=len(wanted):raise ValueError('Some requested rows do not exist')
    return result.loc[requested]


class V2RowReader:
    def __init__(self,root,features):
        self.root=Path(root);self.path=self.root/'data/interim/advanced_mart.parquet'
        schema=set(pq.ParquetFile(self.path).schema_arrow.names)
        self.columns=list(dict.fromkeys(['channel_id','date','target_alarm_next_24h','split']+[c for c in features if c in schema]))
        self.prior=pd.read_parquet(self.root/'data/interim/channel_prior_frozen_v2.parquet')
        self.intra=pd.concat([pd.read_parquet(self.root/f'data/interim/intraday_{year}.parquet') for year in [2025,2026]],ignore_index=True)

    def read(self,indexes):
        result=parquet_rows(self.path,indexes,self.columns).reset_index()
        result=result.merge(self.prior,on='channel_id',how='left',sort=False,validate='many_to_one')
        result=result.merge(self.intra,on=['channel_id','date'],how='left',sort=False,validate='one_to_one')
        return result.set_index('_source_row').loc[indexes]
