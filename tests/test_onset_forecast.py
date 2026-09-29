"""The serving adapter must reject future information and keep its fixed queue."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/serving'))
import object_onset_forecast as onset


class OnsetForecastTests(unittest.TestCase):
    def setUp(self):
        ids = [str(i) for i in range(12)]
        self.meta = pd.DataFrame({'channel_id': ids, 'object_id': ids})
        self.objects = pd.DataFrame({'object_id': ids, 'object_kind': 'kind', 'parent_id': '0',
                                     'диспетчерское_название_объекта': ids})
        def rows(day):
            return pd.DataFrame([{'channel_id': oid, 'date': pd.Timestamp(day),
                **{c: 0 for c in onset.daily.features.COUNTS}, 'events_count': 5,
                'alarm_count': int(oid == '11')} for oid in ids])
        seed = rows('2026-06-25')
        batch = onset.daily.DailyBatch('2026-06-26', '2026-06-27T00:00:00+03:00', rows('2026-06-26'))
        self.history, self.provenance = onset.daily.assemble_history(seed, '2026-06-25', [batch], '2026-06-26')
        self.cfg = {'models': {'hurdle_15': {'sha256': '1'*64}}}

    def produce(self, history=None, provenance=None):
        return onset.predict_payload(self.history if history is None else history,
            self.provenance if provenance is None else provenance, '2026-06-26',
            self.cfg, {}, self.meta, self.objects)

    def test_candidate_cannot_use_december_confirmation_early(self):
        for day in ['2025-11-30', '2025-12-30']:
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, 'unavailable'):
                onset.require_candidate_available(day)
        onset.require_candidate_available('2025-12-31')

    def test_tied_scores_use_string_identity_and_exact_budget(self):
        cards = [{'object_id': str(i), 'score': .4} for i in range(12)]
        self.assertEqual(onset.rank_cards(cards), .4)
        self.assertEqual([c['object_id'] for c in cards], sorted(str(i) for i in range(12)))
        self.assertEqual([c['rank'] for c in cards], list(range(1, 13)))
        self.assertEqual(sum(c['warning'] for c in cards), 10)
        self.assertFalse(cards[-1]['warning'])
        with self.assertRaisesRegex(ValueError, 'Fewer'):
            onset.rank_cards(cards[:9])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            onset.rank_cards(cards[:11]+[cards[0]])

    def test_future_and_unavailable_input_rejected_before_predicting(self):
        future = self.history.copy()
        future.loc[0, 'date'] = pd.Timestamp('2026-06-27')
        invalid = copy.deepcopy(self.provenance)
        invalid['accepted_batches'][0]['available_at'] = '2026-06-27T01:00:00+03:00'
        missing = copy.deepcopy(self.provenance); missing['accepted_batches'] = []
        altered = self.history.copy(); altered.loc[0, 'events_count'] += 1
        with patch.object(onset.experiment, 'predict') as predictor:
            for history, provenance in [(future, self.provenance), (self.history, invalid),
                                        (self.history, missing), (altered, self.provenance)]:
                with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                    self.produce(history, provenance)
            predictor.assert_not_called()

    def test_no_labels_reach_predictor_or_cards_and_current_alarm_not_filtered(self):
        def predict(artifact, frame):
            self.assertFalse(set(onset.FUTURE_COLUMNS).intersection(frame.columns))
            self.assertEqual(len(frame), 12)
            return np.full(len(frame), .4)
        with patch.object(onset.experiment, 'predict', side_effect=predict):
            payload, latest, _ = self.produce()
            repeated, _, _ = self.produce()
        self.assertEqual(payload, repeated)
        self.assertEqual(payload['warnings_count'], 10)
        self.assertEqual(payload['target_kind'], onset.TARGET_KIND)
        self.assertEqual(payload['warning_policy'], onset.POLICY)
        self.assertTrue(next(c for c in payload['cards'] if c['object_id'] == '11')['current_alarm'])
        self.assertTrue(next(c for c in payload['cards'] if c['object_id'] == '11')['warning'])
        self.assertEqual(payload['minimum_lead_hours'], 24)
        self.assertTrue(all(c['explanation_kind'] == 'observations' for c in payload['cards']))
        from api.forecast_store import validate_payload
        validate_payload(payload)

    def test_unexpected_future_label_fails_closed(self):
        original = onset.experiment.onset_frame
        def inject(mart):
            result = original(mart)
            result.loc[result.date.eq('2026-06-26'), 'onset_target'] = 1.
            return result
        with patch.object(onset.experiment, 'onset_frame', side_effect=inject):
            with patch.object(onset.experiment, 'predict') as predictor:
                with self.assertRaisesRegex(ValueError, 'future labels'):
                    self.produce()
                predictor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
