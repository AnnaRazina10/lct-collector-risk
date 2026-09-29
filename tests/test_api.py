import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from api import main

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        folder = Path(self.tmp.name)
        data = folder / 'data.json'
        data.write_text(json.dumps({'cards':[{'id':'sample', 'score':0.2, 'actual_next_day_alarm':True}]}))
        self.patch = patch.multiple(main, DATA=data, DB=folder / 'tickets.sqlite3')
        self.patch.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        self.patch.stop()
        self.tmp.cleanup()

    def test_future_outcome_is_separate(self):
        self.assertNotIn('actual_next_day_alarm',self.client.get('/api/risks').json()['cards'][0])
        self.assertTrue(self.client.get('/api/risks/sample/outcome').json()['actual_next_day_alarm'])
        self.assertEqual(self.client.get('/api/risks/unknown/outcome').status_code,404)

    def test_draft_is_local_persistent_and_idempotent(self):
        first=self.client.post('/api/tickets',json={'risk_id':'sample','note':"Проверить 'канал'"})
        self.assertEqual(first.status_code,201)
        self.assertFalse(first.json()['external_submission'])
        second=self.client.post('/api/tickets',json={'risk_id':'sample'})
        self.assertEqual(first.json()['id'],second.json()['id'])
        tickets=self.client.get('/api/tickets').json()['tickets']
        self.assertEqual(len(tickets),1)
        self.assertEqual(tickets[0]['note'],"Проверить 'канал'")

    def test_reject_invalid_drafts(self):
        self.assertEqual(self.client.post('/api/tickets',json={'risk_id':'unknown'}).status_code,404)
        self.assertEqual(self.client.post('/api/tickets',json={'risk_id':'sample','note':'a'*2001}).status_code,422)

if __name__ == '__main__':
    unittest.main()
