import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/modeling'))
from feature_rows import parquet_rows


class RowReaderTests(unittest.TestCase):
    def test_sparse_order_categories_and_nan_survive_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.parquet'
            source=pd.DataFrame({'x':[0.,np.nan,2.,3.,4.,5.,6.],
                                 'cat':pd.Categorical(['a','b','a','b','c','a','c'])})
            source.to_parquet(path,index=False)
            indexes=[6,1,4,0]
            got=parquet_rows(path,indexes,batch_size=2)
            expected=source.loc[indexes].copy();expected.index.name='_source_row'
            pd.testing.assert_frame_equal(got,expected)

    def test_rejects_out_of_range_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.parquet';pd.DataFrame({'x':[1,2]}).to_parquet(path,index=False)
            with self.assertRaises(ValueError):parquet_rows(path,[0,9])

if __name__=='__main__':unittest.main()
