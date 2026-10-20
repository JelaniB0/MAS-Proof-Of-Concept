from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentbalance.config import load_config
from agentbalance.files import write_json
from agentbalance.pipeline import execute_stage
from agentbalance.providers import Completion, ProviderError
from agentbalance.research import main as research_main

ROOT=Path(__file__).resolve().parents[2]


class PipelineCliRecoveryTests(unittest.TestCase):
    def test_real_training_cli_resumes_saved_update_after_provider_failure(self):
        import torch
        from agentbalance.adaptive.controller import build_controller, load_controller
        from agentbalance.adaptive.encoder import TextEncoder
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root=Path(directory)
            config,data,_,_=self.prepare(root)
            settings=replace(load_config(config),agent_templates={'analyst':'Solve it.','synthesizer':'Answer.'})
            write_json(config,asdict(settings))
            data.write_text(''.join(json.dumps({'task':f'Choose A question {i}', 'answers':['A'],
                                               'scorer':'mmlu'})+'\n' for i in range(2)))
            initial=root/'initial.pt'
            build_controller(settings,TextEncoder('hash-test',16),hidden=16).save(initial)
            output=root/'trained.pt'
            index={'calls':0,'accounted_usd':0.,'stages':[]}
            flags=['--checkpoint',initial,'--epochs','1','--batch-size','1','--topology','star',
                   '--cost-weight','0','--latency-weight','0','--seed','42','--no-env']
            allowance=lambda:['--max-calls','10','--total-budget','1','--budget','1','--latency-budget','120']
            with patch('agentbalance.pipeline.invoke',side_effect=self.invoke_cli), patch(
                    'agentbalance.providers.complete',side_effect=[Completion('A',5,5,'provider','stop'),
                    Completion('A',5,5,'provider','stop'),ProviderError('HTTP 429')]):
                with self.assertRaises(ValueError):
                    execute_stage(index,root/'index.json','train',config,data,output,flags,allowance)
            original=output.read_bytes()
            report=json.loads(output.with_suffix('.training.json').read_text())
            self.assertEqual(report['calls'],3)
            self.assertEqual(len(report['runs']),1)
            state=torch.load(output.with_suffix('.optimizer.pt'),weights_only=True)
            self.assertEqual(state['next_index'],1)
            with patch('agentbalance.pipeline.invoke',side_effect=self.invoke_cli), patch(
                    'agentbalance.providers.complete',return_value=Completion('A',5,5,'provider','stop')) as complete:
                execute_stage(index,root/'index.json','train',config,data,output,flags,allowance,
                              resume_training=True)
            self.assertEqual(complete.call_count,2)
            self.assertEqual(index['calls'],5)
            self.assertEqual((root/'trained.failed-001.pt').read_bytes(),original)
            recovered=json.loads(output.with_suffix('.training.json').read_text())
            self.assertEqual(recovered['status'],'completed')
            self.assertEqual(len(recovered['runs']),1)
            self.assertNotEqual(recovered['runs'][0]['key'],report['runs'][0]['key'])
            self.assertEqual(load_controller(output,settings,allow_untrained=True).training_info['policy_steps'],2)

    def test_profile_resume_reuses_successes_and_accounts_only_new_attempts(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root=Path(directory)
            config,data,_,_=self.prepare(root)
            output=root/'measured.json'
            index={'calls':0,'accounted_usd':0.,'stages':[]}
            flags=['--no-env']
            allowance=lambda:['--max-calls','10','--total-budget','1','--budget','1','--latency-budget','120']
            with patch('agentbalance.pipeline.invoke',side_effect=self.invoke_cli), patch(
                    'agentbalance.providers.complete',side_effect=[Completion('A',5,5,'provider','stop'),
                                                                 ProviderError('quota')]) as complete:
                with self.assertRaises(ValueError):
                    execute_stage(index,root/'index.json','profile',config,data,output,flags,allowance)
            self.assertEqual(complete.call_count,2)
            self.assertEqual(index['calls'],2)
            trace=output.with_suffix('.measurements.jsonl')
            original=trace.read_bytes()
            # Another provider failure is charged once, and retains the saved success.
            with patch('agentbalance.pipeline.invoke',side_effect=self.invoke_cli), patch(
                    'agentbalance.providers.complete',side_effect=ProviderError('quota')):
                with self.assertRaises(ValueError):
                    execute_stage(index,root/'index.json','profile',config,data,output,flags,allowance,
                                  resume_profiles=True)
            self.assertEqual(index['calls'],3)
            self.assertTrue(trace.read_bytes().startswith(original))
            # Exactly the missing model/task outcome fits in this allowance.
            allowance=lambda:['--max-calls','1','--total-budget','1','--budget','1','--latency-budget','120']
            with patch('agentbalance.pipeline.invoke',side_effect=self.invoke_cli), patch(
                    'agentbalance.providers.complete',return_value=Completion('A',5,5,'provider','stop')) as complete:
                execute_stage(index,root/'index.json','profile',config,data,output,flags,allowance,
                              resume_profiles=True)
                execute_stage(index,root/'index.json','profile',config,data,output,flags,allowance,
                              resume_profiles=True)
            self.assertEqual(complete.call_count,1)
            self.assertEqual(index['calls'],4)
            self.assertTrue(trace.read_bytes().startswith(original))
            rows=[json.loads(line) for line in trace.read_text().splitlines()]
            self.assertAlmostEqual(index['accounted_usd'],sum(
                row['cost_usd'] if row['cost_usd'] is not None else row['reserved_cost_usd'] for row in rows))
            self.assertTrue(all(model.quality == 1. for model in load_config(output).models))

    def test_profile_resume_retries_incomplete_and_grader_failed_results(self):
        for completion,status in ((Completion('',5,5,'provider','length'),None),
                                  (Completion('A',5,5,'provider','stop'),'grader_error')):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
                from agentbalance.adaptive.data import load_records
                root=Path(directory)
                config,data,_,_=self.prepare(root)
                output=root/'measured.json'
                trace=output.with_suffix('.measurements.jsonl')
                item={'key':load_records(data)[0]['key'],'model_id':load_config(config).models[0].id,
                      'completion':asdict(completion),'score':None if status else 0.,
                      'cost_usd':.1,'latency_seconds':1.}
                if status:
                    item['status']=status
                trace.write_text(json.dumps(item)+'\n')
                index={'calls':1,'accounted_usd':.1,'stages':[]}
                with patch('agentbalance.pipeline.invoke',side_effect=self.invoke_cli), patch(
                        'agentbalance.providers.complete',return_value=Completion('A',5,5,'provider','stop')) as complete:
                    execute_stage(index,root/'index.json','profile',config,data,output,['--no-env'],
                                  lambda:['--max-calls','2','--total-budget','1'],resume_profiles=True)
                self.assertEqual(complete.call_count,2)
                self.assertEqual(index['calls'],3)
                self.assertTrue(all(model.quality == 1. for model in load_config(output).models))

    def test_profile_resume_preflight_rejection_does_not_recount_trace(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root=Path(directory)
            config,data,_,_=self.prepare(root)
            output=root/'measured.json'
            trace=output.with_suffix('.measurements.jsonl')
            trace.write_text(json.dumps({'key':'wrong-task','model_id':'cheap','status':'error',
                                         'cost_usd':None,'reserved_cost_usd':.2})+'\n')
            original=trace.read_bytes()
            index={'calls':1,'accounted_usd':.2,'stages':[]}
            with patch('agentbalance.pipeline.invoke',side_effect=self.invoke_cli), patch(
                    'agentbalance.providers.complete') as complete:
                with self.assertRaises(ValueError):
                    execute_stage(index,root/'index.json','profile',config,data,output,['--no-env'],
                                  lambda:['--max-calls','10','--total-budget','1'],resume_profiles=True)
            complete.assert_not_called()
            self.assertEqual(index['calls'],1)
            self.assertEqual(index['accounted_usd'],.2)
            self.assertEqual(trace.read_bytes(),original)

    def prepare(self,root):
        config=root/'config.json'
        write_json(config,asdict(load_config(ROOT/'agentbalance/demo.json')))
        data=root/'data.jsonl'
        data.write_text(json.dumps({'task':'Choose A','answers':['A'],'scorer':'mmlu'})+'\n',encoding='utf-8')
        output=root/'evaluation.json'
        write_json(output,{'status':'running','calls':6,'accounted_usd':0.,'runs':[]})
        return config,data,output,{'calls':6,'accounted_usd':0.,'stages':[]}

    def invoke_cli(self,arguments):
        code=research_main([str(a) for a in arguments])
        if code:
            raise ValueError(f'CLI failed with exit {code}')

    def test_real_evaluation_cli_retries_partial_output_and_then_skips_completed_stage(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root=Path(directory)
            config,data,output,index=self.prepare(root)
            previous=output.read_bytes()
            flags=['--policy','cheap','--topology','star','--seed','42','--no-env']
            allowance=lambda:['--max-calls','10','--total-budget','1','--budget','1','--latency-budget','120']
            with patch('agentbalance.pipeline.invoke',side_effect=self.invoke_cli) as invoke, patch(
                    'agentbalance.providers.complete',return_value=Completion('A',5,5,'provider','stop')) as complete:
                execute_stage(index,root/'index.json','evaluate',config,data,output,flags,allowance)
                execute_stage(index,root/'index.json','evaluate',config,data,output,flags,allowance)
            self.assertEqual(invoke.call_count,1)
            self.assertEqual(complete.call_count,3)
            self.assertEqual(index['calls'],9)
            report=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['accuracy'],1.)
            self.assertEqual((root/'evaluation.failed-001.json').read_bytes(),previous)

    def test_actual_cli_preflight_rejection_adds_no_calls(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root=Path(directory)
            config,data,output,index=self.prepare(root)
            settings=json.loads(config.read_text(encoding='utf-8'))
            settings['max_output_tokens']='invalid'
            write_json(config,settings)
            with patch('agentbalance.pipeline.invoke',side_effect=self.invoke_cli), patch(
                    'agentbalance.providers.complete') as complete:
                with self.assertRaises(ValueError):
                    execute_stage(index,root/'index.json','evaluate',config,data,output,
                                  ['--policy','cheap','--seed','42','--no-env'],
                                  lambda:['--max-calls','10','--total-budget','1'])
            complete.assert_not_called()
            self.assertEqual(index['calls'],6)
            self.assertEqual(index['stages'],[])
            self.assertTrue((root/'evaluation.failed-001.json').exists())


if __name__=='__main__':
    unittest.main()
