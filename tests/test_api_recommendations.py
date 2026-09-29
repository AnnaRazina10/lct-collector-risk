"""Current review advice is separate from immutable forecasts and server-derived."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import forecast_store, main
from tests.test_forecast_store import payload
from tests.test_forecast_onset_policy import onset_payload


def full_cards(value):
    for card in value['cards']:
        card['id'] = value['run_id']+'_'+card['object_id']
        card.update(observed_channels=10, catalog_channels=10, events_today=100,
                    alarm_channels=0, current_alarm=False, alarm_days_7d=0, observed_days_7d=7)
    return value


class RecommendationsApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db = Path(self.folder.name)/'archive.sqlite3'
        self.onset = full_cards(onset_payload())
        self.onset['cards'][0].update(observed_channels=0, events_today=0, observed_days_7d=0)
        self.onset['cards'][1].update(observed_channels=6, alarm_channels=2, current_alarm=True, alarm_days_7d=3)
        self.any_alarm = full_cards(payload())
        for value in [self.onset, self.any_alarm]: forecast_store.publish_run(self.db, value)
        self.original_metadata = forecast_store.list_runs(self.db)
        legacy = Path(self.folder.name)/'legacy.json'
        legacy.write_text(json.dumps({'cards': [{'id': 'legacy', 'score': .2, 'actual_next_day_alarm': True}]}))
        self.patcher = patch.multiple(main, DB=self.db, OBJECT_DATA=legacy, DATA=legacy)
        self.patcher.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close(); self.patcher.stop(); self.folder.cleanup()

    def query(self, value):
        return '?mode=object&run_id='+value['run_id']

    def identity(self, value, position=0):
        return {'risk_id': value['cards'][position]['id'], 'entity_mode': 'object', 'run_id': value['run_id']}

    def test_read_advice_preserves_all_forecasts_top10_and_does_not_save_decisions(self):
        response = self.client.get('/api/risks'+self.query(self.onset))
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['cards'], self.onset['cards'])
        self.assertEqual(sum(card['warning'] for card in result['cards']), 10)
        suggestions = result['recommendations']
        self.assertEqual(suggestions[self.onset['cards'][0]['id']]['rule_ids'], ['NO_RECORDS_TODAY'])
        self.assertEqual(suggestions[self.onset['cards'][1]['id']]['rule_ids'],
                         ['PARTIAL_CATALOG_RECORDS', 'ALARM_RECORDED_TODAY', 'MULTIPLE_ALARM_DAYS'])
        self.assertTrue(all(r['status'] == 'ok' for r in suggestions.values()))
        self.assertTrue(all(r['generated_at'] > self.onset['issue_time'] for r in suggestions.values()))
        self.assertEqual(forecast_store.load_run(self.db, self.onset['run_id']), self.onset)
        self.assertEqual(forecast_store.list_runs(self.db), self.original_metadata)
        with sqlite3.connect(self.db) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn('feedback', tables); self.assertNotIn('tickets', tables)

    def test_both_goals_have_distinct_deterministic_identity_and_snapshot(self):
        identities = []
        for value, goal in [(self.onset, 'registered_episode_start_g1'), (self.any_alarm, 'any_alarm')]:
            card = value['cards'][0]
            path = '/api/risks/'+card['id']+self.query(value)
            a, b = self.client.get(path).json(), self.client.get(path).json()
            self.assertEqual(a['card'], card)
            self.assertEqual(a['recommendation']['goal'], goal)
            self.assertEqual(a['recommendation']['recommendation_id'], b['recommendation']['recommendation_id'])
            identities.append(a['recommendation']['recommendation_id'])
        self.assertNotEqual(*identities)

    def test_feedback_snapshot_is_server_derived_and_survives_new_client_and_rules(self):
        identity = self.identity(self.onset, 1)
        response = self.client.post('/api/feedback', json={**identity, 'decision': 'inspect',
            'reason': 'needs_inspection', 'operator': 'ТЕСТ', 'note': 'Локальная проверка',
            'recommendation': {'recommendation_id': 'forged'}, 'facts': {'current_alarm': False}})
        self.assertEqual(response.status_code, 201)
        saved = response.json()['risk_snapshot']['recommendation']
        self.assertNotEqual(saved['recommendation_id'], 'forged')
        self.assertIn('ALARM_RECORDED_TODAY', saved['rule_ids'])
        with patch.object(main, 'recommendation', return_value={'status': 'changed-later'}):
            with TestClient(main.app) as later:
                history = later.get('/api/journal?mode=object').json()['entries']
        self.assertEqual(history[0]['risk_snapshot']['recommendation'], saved)
        self.assertEqual(forecast_store.list_runs(self.db), self.original_metadata)

    def test_existing_ticket_keeps_original_recommendation_and_wrong_run_rejected(self):
        identity = self.identity(self.onset)
        first = self.client.post('/api/tickets', json=identity).json()
        with patch.object(main, 'recommendation', return_value={'status': 'changed-later'}):
            second = self.client.post('/api/tickets', json=identity).json()
        self.assertTrue(second['already_exists'])
        self.assertEqual(first['risk_snapshot'], second['risk_snapshot'])
        wrong = {**identity, 'run_id': self.any_alarm['run_id']}
        self.assertEqual(self.client.post('/api/tickets', json=wrong).status_code, 404)

    def test_incomplete_old_archive_fails_advice_closed_and_legacy_modes_remain_readable(self):
        incomplete = payload('object_2026-06-28_incomplete')
        forecast_store.publish_run(self.db, incomplete)
        response = self.client.get('/api/risks'+self.query(incomplete)).json()
        advice = response['recommendations'][incomplete['cards'][0]['id']]
        self.assertEqual(advice['status'], 'unavailable'); self.assertEqual(advice['steps'], [])
        for mode in ['channel', 'object']:
            legacy = self.client.get('/api/risks?mode='+mode).json()
            self.assertNotIn('recommendations', legacy)
            self.assertNotIn('actual_next_day_alarm', legacy['cards'][0])


if __name__ == '__main__':
    unittest.main()
