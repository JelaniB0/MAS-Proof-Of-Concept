import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agentbalance.files import write_json, replace_file
from agentbalance.pipeline import execute_stage


class PipelineRecoveryTests(unittest.TestCase):
    def test_interrupted_training_archive_is_finished_on_next_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/'trained.pt'
            output.write_bytes(b'checkpoint')
            output.with_suffix('.optimizer.pt').write_bytes(b'optimizer')
            write_json(output.with_suffix('.training.json'),{'status':'error','calls':20})
            index={'calls':20,'accounted_usd':0.,'stages':[]}
            calls=0
            def interrupted_replace(source,target):
                nonlocal calls
                calls+=1
                if calls == 2:
                    raise PermissionError('File temporarily locked')
                return replace_file(source,target)
            with patch('agentbalance.pipeline.replace_file',side_effect=interrupted_replace), patch(
                    'agentbalance.pipeline.invoke') as invoke:
                with self.assertRaises(PermissionError):
                    execute_stage(index,root/'index.json','train','config','data',output,[],lambda:[],
                                  resume_training=True)
            invoke.assert_not_called()
            self.assertEqual(index['calls'],20)
            self.assertEqual(index['training_retries'][0]['status'],'archiving')
            def complete(arguments):
                archived=root/'trained.failed-001.pt'
                self.assertEqual(archived.with_suffix('.optimizer.pt').read_bytes(),b'optimizer')
                self.assertEqual(arguments[arguments.index('--checkpoint')+1],archived.resolve())
                output.write_bytes(b'recovered')
                write_json(output.with_suffix('.training.json'),{'status':'completed','calls':4})
            with patch('agentbalance.pipeline.invoke',side_effect=complete):
                execute_stage(index,root/'index.json','train','config','data',output,[],lambda:[],
                              resume_training=True)
            self.assertEqual(index['calls'],24)
            self.assertEqual(index['training_retries'][0]['status'],'ready')

    def test_training_resume_archives_state_and_retries_after_cli_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/'trained.pt'
            output.write_bytes(b'checkpoint')
            output.with_suffix('.optimizer.pt').write_bytes(b'optimizer')
            output.with_suffix('.rollouts.jsonl').write_bytes(b'rollout')
            write_json(output.with_suffix('.training.json'),{'status':'provider_or_budget_failure',
                                                          'calls':20,'accounted_usd':.2})
            index={'calls':20,'accounted_usd':.2,'stages':[]}
            flags=['--checkpoint',root/'initial.pt','--epochs','1']
            with patch('agentbalance.pipeline.invoke',side_effect=ValueError('Preflight rejected')) as invoke:
                with self.assertRaises(ValueError):
                    execute_stage(index,root/'index.json','train','config','data',output,flags,lambda:[],
                                  resume_training=True)
            archived=root/'trained.failed-001.pt'
            self.assertEqual(archived.read_bytes(),b'checkpoint')
            self.assertEqual(archived.with_suffix('.optimizer.pt').read_bytes(),b'optimizer')
            self.assertEqual(index['calls'],20)
            self.assertEqual(index['accounted_usd'],.2)
            command=invoke.call_args.args[0]
            self.assertEqual(command[command.index('--checkpoint')+1],archived.resolve())
            self.assertIn('--resume-training',command)
            # Recovery remains required even when the rejected CLI wrote no new artifacts.
            with patch('agentbalance.pipeline.invoke') as invoke:
                with self.assertRaisesRegex(ValueError,'--resume-training'):
                    execute_stage(index,root/'index.json','train','config','data',output,flags,lambda:[])
                invoke.assert_not_called()
            def complete(arguments):
                self.assertEqual(arguments[arguments.index('--checkpoint')+1],archived.resolve())
                output.write_bytes(b'recovered checkpoint')
                write_json(output.with_suffix('.training.json'),{'status':'completed','calls':6,'accounted_usd':.05})
            with patch('agentbalance.pipeline.invoke',side_effect=complete) as invoke:
                execute_stage(index,root/'index.json','train','config','data',output,flags,lambda:[],
                              resume_training=True)
                execute_stage(index,root/'index.json','train','config','data',output,flags,lambda:[],
                              resume_training=True)
            self.assertEqual(invoke.call_count,1)
            self.assertEqual(index['calls'],26)
            self.assertAlmostEqual(index['accounted_usd'],.25)
            self.assertEqual(archived.read_bytes(),b'checkpoint')

    def test_training_recovery_requires_optimizer_and_rejects_completed_unrecorded_stage(self):
        for completed in (False,True):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                output=root/'trained.pt'
                output.write_bytes(b'checkpoint')
                if completed:
                    output.with_suffix('.optimizer.pt').write_bytes(b'optimizer')
                    write_json(output.with_suffix('.training.json'),{'status':'completed','calls':3})
                index={'calls':3,'accounted_usd':0.,'stages':[]}
                with patch('agentbalance.pipeline.invoke') as invoke:
                    with self.assertRaisesRegex(ValueError,'Completed training' if completed else 'optimizer'):
                        execute_stage(index,root/'index.json','train','config','data',output,[],lambda:[],
                                      resume_training=True)
                invoke.assert_not_called()
                self.assertEqual(output.read_bytes(),b'checkpoint')
                self.assertEqual(index['calls'],3)

    def test_rejected_retry_does_not_count_previous_attempt_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/'evaluation.json'
            write_json(output,{'status':'running','calls':12,'accounted_usd':.1,'note':'saved question — unchanged'})
            write_json(output.with_suffix('.json.tmp'),{'status':'running','calls':12,'accounted_usd':.1})
            original=output.read_bytes()
            index={'calls':12,'accounted_usd':.1,'stages':[]}
            with patch('agentbalance.pipeline.invoke',side_effect=ValueError('CLI rejected the retry')):
                with self.assertRaises(ValueError):
                    execute_stage(index,root/'index.json','evaluate','config','data',output,[],lambda:[])
            self.assertEqual(index['calls'],12)
            self.assertEqual(index['accounted_usd'],.1)
            self.assertEqual((root/'evaluation.failed-001.json').read_bytes(),original)
            self.assertTrue((root/'evaluation.failed-001.json.tmp').exists())
            self.assertFalse(output.exists())

    def test_failed_save_accounts_current_temporary_snapshot_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/'evaluation.json'
            write_json(output,{'status':'running','calls':12,'accounted_usd':.1})
            index={'calls':12,'accounted_usd':.1,'stages':[]}
            def fail_after_requests(arguments):
                write_json(output,{'status':'running','calls':3,'accounted_usd':.2})
                write_json(output.with_suffix('.json.tmp'),{'status':'running','calls':6,'accounted_usd':.4})
                raise PermissionError('Final replacement failed')
            with patch('agentbalance.pipeline.invoke',side_effect=fail_after_requests):
                with self.assertRaises(PermissionError):
                    execute_stage(index,root/'index.json','evaluate','config','data',output,[],lambda:[])
            self.assertEqual(index['calls'],18)
            self.assertAlmostEqual(index['accounted_usd'],.5)
            self.assertEqual(index['stages'],[])

    def test_completed_report_without_ledger_entry_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/'evaluation.json'
            write_json(output,{'status':'completed','calls':12})
            original=output.read_bytes()
            index={'calls':12,'accounted_usd':0.,'stages':[]}
            with patch('agentbalance.pipeline.invoke') as invoke:
                with self.assertRaisesRegex(ValueError,'Completed evaluation'):
                    execute_stage(index,root/'index.json','evaluate','config','data',output,[],lambda:[])
            invoke.assert_not_called()
            self.assertEqual(output.read_bytes(),original)
            self.assertEqual(index['calls'],12)

    def test_cached_stage_is_skipped_even_when_no_allowance_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/'evaluation.json'
            write_json(output,{'status':'completed','calls':12})
            index={'calls':12,'accounted_usd':0.,'stages':[str(output.resolve())]}
            allowance=Mock(side_effect=ValueError('No allowance'))
            with patch('agentbalance.pipeline.invoke') as invoke:
                execute_stage(index,root/'index.json','evaluate','config','data',output,[],allowance)
            invoke.assert_not_called()
            allowance.assert_not_called()
            self.assertEqual(index['calls'],12)

    def test_incomplete_training_and_profiling_are_rejected_before_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for kind in ('train','profile'):
                output=root/(kind+'.json')
                write_json(output,{'calls':12})
                index={'calls':12,'accounted_usd':0.,'stages':[]}
                with patch('agentbalance.pipeline.invoke') as invoke:
                    with self.assertRaisesRegex(ValueError,'explicit recovery'):
                        execute_stage(index,root/'index.json',kind,'config','data',output,[],lambda:[])
                invoke.assert_not_called()
                self.assertEqual(index['calls'],12)


if __name__=='__main__':
    unittest.main()
