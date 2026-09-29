"""Check the standalone CLI against saved calibration scores without labels."""
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import experiment_v3 as v3
from feature_rows import V2RowReader, parquet_rows


def main():
    cfg = json.loads((v3.OUT / 'selection.json').read_text())
    features = []

    def visit(name):
        node = cfg['configs'][name]
        if node['kind'] in ['blend', 'product']:
            for part in node['parts']:
                visit(part)
        else:
            features.extend(node['features'])

    visit(cfg['selected_model'])
    features = list(dict.fromkeys(features))
    keys = pd.read_parquet(v3.v2.CACHE, columns=['channel_id', 'date'])
    calibration_indexes = keys.index[(keys.date >= '2025-12-01') & (keys.date <= '2025-12-30')]
    indexes = calibration_indexes[np.linspace(0, len(calibration_indexes) - 1, 256, dtype=int)]
    frame = V2RowReader(v3.ROOT, features).read(indexes)
    sequence = parquet_rows(v3.CACHE, indexes)
    pd.testing.assert_frame_equal(frame[['channel_id', 'date']], sequence[['channel_id', 'date']], check_names=False)
    for column in features:
        if column not in frame:
            frame[column] = sequence[column]
    frame = frame[['channel_id', 'date'] + features]
    assert v3.TARGET not in frame and v3.OBS not in frame
    input_path = v3.OUT / 'predictions/inference_features.parquet'
    output_path = v3.OUT / 'predictions/inference_output.csv'
    frame.to_parquet(input_path, index=False)
    subprocess.run([sys.executable, str(v3.ROOT / 'src/modeling/predict_risk_v3.py'),
                    '--features', str(input_path), '--output', str(output_path)], check=True)
    actual = pd.read_csv(output_path, parse_dates=['date'], dtype={'channel_id': 'str'})
    expected = pd.read_parquet(v3.OUT / 'predictions/calibration.parquet')
    checked = actual.merge(expected, on=['channel_id', 'date'], validate='one_to_one')
    assert len(checked) == 256
    difference = float(np.abs(checked.score_x - checked.score_y).max())
    if difference > 1e-12:
        raise ValueError('Standalone predictions differ from calibration')
    np.testing.assert_array_equal(checked.warning, checked.score_y >= cfg['threshold'])
    v3.dump(v3.OUT / 'inference_cli_check.json', {
        'rows': len(checked), 'maximum_difference': difference,
        'labels_in_input': False, 'warning_match': True, 'passed': True,
        'selection_sha256': v3.sha(v3.OUT / 'selection.json')})
    print('Independent CLI reproduction passed:', difference)


if __name__ == '__main__':
    main()
