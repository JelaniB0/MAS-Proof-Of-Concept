from dataclasses import replace
from pathlib import Path
import json
import copy
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import torch

from agentbalance.adaptive.controller import build_controller, load_controller
from agentbalance.adaptive.data import task_key
from agentbalance.adaptive.encoder import TextEncoder
from agentbalance.adaptive.metrics import frontier, auc, performance_at_budget, measurement, summarize
from agentbalance.adaptive.pools import construct_pools
from agentbalance.adaptive.scoring import Scorer
from agentbalance.adaptive.training import RunAllowance, train_policy, evaluate, evaluate_baseline
from agentbalance.config import load_config
from agentbalance.difficulty_training import domain_weights
from agentbalance.graph import dag_waves
from agentbalance.paper_data import prepare_mmlu, prepare_humaneval
from agentbalance.providers import Completion
from agentbalance.runner import execute
from agentbalance.planner import Plan

ROOT = Path(__file__).resolve().parents[2]


class PaperTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        base = load_config(ROOT/'agentbalance/demo.json')
        self.config = replace(base, controller_method='paper', min_workers=2, scheduler='ready',
                              decision_model=base.strong_model,
                              agent_templates={**{f'worker{i}': f'Reason about topic {i}' for i in range(5)},
                                               'synthesizer': 'Give the answer.'})
        self.controller = build_controller(self.config, TextEncoder('hash-test', 16, normalize=False), hidden=16)

    def test_paper_graphs_and_decision_node(self):
        for _ in range(15):
            plan, logp, entropy, penalty = self.controller.select('Question', sample=True)
            dag_waves(plan.assignments, plan.edges)
            self.assertGreaterEqual(len(plan.assignments), 3)
            self.assertEqual(plan.assignments['synthesizer'], self.config.strong_model)
            self.assertNotIn('synthesizer', plan.selection['keep_probabilities'])
            self.assertNotIn('synthesizer', plan.selection['model_probabilities'])
            self.assertGreaterEqual(plan.selection['hop_limit'], 2.)
            self.assertTrue(torch.isfinite(logp+entropy+penalty))

    def test_offset_upper_bound_and_ease_orientation(self):
        config = replace(self.config, explicit_pools=(('cheap',), ('strong',)), pool_upper_bound=0,
                         difficulty_offset=.3)
        controller = build_controller(config, TextEncoder('hash-test', 16), hidden=16)
        plan, *_ = controller.select('Question')
        self.assertEqual(plan.selection['selected_pool'], 0)
        self.assertAlmostEqual(plan.difficulty, 1-plan.selection['ease'], places=6)
        self.assertAlmostEqual(plan.selection['effective_difficulty'], min(1., plan.difficulty+.3), places=6)

    def test_paper_gradients_and_hop_penalty(self):
        with torch.no_grad():
            self.controller.gate.weight.zero_()
            self.controller.gate.bias.fill_(10)
            self.controller.edge_projection.weight.zero_()
            self.controller.edge_projection.bias.fill_(1)
            self.controller.hops.weight.zero_()
            self.controller.hops.bias.copy_(torch.tensor([8., -8., -8., -8.]))
        _, logp, entropy, penalty = self.controller.select('Question', sample=True)
        (-logp+.2*penalty).backward()
        self.assertGreater(float(penalty.detach()), 0.)
        for module in (self.controller.difficulty, self.controller.role_match, self.controller.context_output,
                       self.controller.gate, self.controller.hops):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_ablation_no_gating_and_complete_topology(self):
        config = replace(self.config, ablation='no-gating')
        c = build_controller(config, TextEncoder('hash-test',16), hidden=16)
        self.assertEqual(len(c.plan('Question', budget_usd=1).assignments), 6)
        config = replace(self.config, ablation='no-topology')
        c = build_controller(config, TextEncoder('hash-test',16), hidden=16)
        plan = c.plan('Question', budget_usd=1)
        n = len(plan.assignments)-1
        self.assertEqual(len(plan.edges), n*(n-1)//2+n)

    def test_checkpoint_roundtrip_and_local_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trained.pt'
            rows = [{'task': 'Return 42', 'answers': ['42'], 'key': task_key('Return 42')}]
            with patch('agentbalance.providers.complete', return_value=Completion('42', 5, 5, 'provider', 'stop')):
                result = train_policy(self.controller, rows, path, RunAllowance(self.config, 30, 1.),
                                      epochs=2, reward_units='raw', length_weight=.2)
            self.assertEqual(result['status'], 'completed')
            restored = load_controller(path, self.config, allow_untrained=True)
            self.assertEqual(restored.training_info['policy_steps'], 2)
            self.controller.eval()
            self.assertEqual(restored.plan('Question', budget_usd=1).assignments,
                             self.controller.plan('Question', budget_usd=1).assignments)
            self.assertEqual(torch.load(path, weights_only=True)['format'], 3)

    def test_ready_scheduler_releases_successor_without_wave_barrier(self):
        config = replace(self.config, agent_templates={'fast':'F','slow':'S','child':'C','synthesizer':'D'})
        plan = Plan('adaptive','learned', {r:'cheap' for r in config.agent_templates},0,1,0,.5,1,2,
                    (('fast','child'),('child','synthesizer'),('slow','synthesizer')), config.agent_templates)
        child_started = threading.Event()
        def complete(model, prompt, *args):
            role = prompt[0]['content']
            if role == 'S':
                self.assertTrue(child_started.wait(1))
            elif role == 'C':
                child_started.set()
            return Completion('42',5,5,'provider','stop')
        with patch('agentbalance.providers.complete', side_effect=complete):
            report = execute(config,plan,'Question')
        self.assertEqual(report['status'],'completed')
        self.assertEqual(report['totals']['calls'],4)

    def test_ready_scheduler_admission_checks_all_running_reservations(self):
        plan = replace(self.controller.plan('Question', budget_usd=1), budget_usd=0.)
        config = replace(self.config, models=tuple(replace(m,input_usd_per_million=100) for m in self.config.models))
        with patch('agentbalance.providers.complete') as call:
            report = execute(config, plan, 'Question')
        self.assertEqual(report['status'], 'budget_stopped')
        call.assert_not_called()

    def test_balance_has_expected_domain_mass(self):
        rows = [{'eval_names':[g]} for g in ['mbpp']*3+['gsm8k']*2+['mmlu']*8]
        weights, counts = domain_weights(rows)
        self.assertAlmostEqual(sum(weights[:3]),1.)
        self.assertAlmostEqual(sum(weights[3:5]),1.)
        self.assertAlmostEqual(sum(weights[5:]),4.)
        with self.assertRaises(ValueError):
            domain_weights(rows[:3])

    def test_large_pool_downsamples_farthest_members(self):
        template=self.config.get('strong')
        models=tuple(replace(template,id=str(i),quality=i/10,
                     input_usd_per_million=i+1,latency_seconds=i+1) for i in range(6))
        full,_=construct_pools(models,count=1,min_size=1)
        bounded,_=construct_pools(models,count=1,min_size=1,max_size=2)
        self.assertEqual(len(full[0]),6)
        self.assertEqual(len(bounded[0]),2)
        self.assertNotIn('5',bounded[0])

    def test_tool_failure_preserves_known_provider_usage(self):
        config = replace(self.config, agent_tools='math', agent_tool_image='python:3.13-slim',
                         agent_templates={'programming_expert':'P','synthesizer':'D'})
        plan = Plan('star','learned',{'programming_expert':'strong','synthesizer':'strong'},0,1,0,.5,1,120,
                    (('programming_expert','synthesizer'),),config.agent_templates)
        with patch.object(Scorer,'preflight'), patch('agentbalance.providers.complete',
                return_value=Completion('answer = 42',5,5,'provider','stop')), patch(
                'agentbalance.agent_tools.run_code',side_effect=ValueError('Container failed')):
            report = execute(config,plan,'Question')
        self.assertEqual(report['status'],'error')
        self.assertEqual(report['totals']['calls'],1)
        self.assertTrue(report['totals']['cost_complete'])
        self.assertEqual(report['totals']['cost_usd'],config.get('strong').cost(5,5))

    def test_public_tool_failure_before_inference_counts_zero_calls(self):
        config = replace(self.config, agent_tools='humaneval', agent_tool_image='python:3.13-slim')
        plan = self.controller.plan('Question',budget_usd=1)
        with patch.object(Scorer,'preflight'), patch('agentbalance.providers.complete') as call, patch(
                'agentbalance.agent_tools.check_contributions',side_effect=ValueError('Container failed')):
            report=execute(config,plan,'Question')
        call.assert_not_called()
        self.assertEqual(report['totals']['calls'],0)
        self.assertTrue(report['totals']['cost_complete'])

    def test_grader_failure_preserves_evaluation_usage(self):
        rows=[{'task':'Question','key':task_key('Question'),'answers':['42']}]
        scorer=Scorer()
        with tempfile.TemporaryDirectory() as directory, patch('agentbalance.providers.complete',
                return_value=Completion('42',5,5,'provider','stop')), patch.object(
                scorer,'score',side_effect=ValueError('Grader stopped')):
            for learned in (False,True):
                out=Path(directory)/f'{learned}.json'
                with self.assertRaises(ValueError):
                    if learned:
                        evaluate(self.controller,rows,out,RunAllowance(self.config,50,1),budget=1,scorer=scorer)
                    else:
                        evaluate_baseline(self.config,rows,out,RunAllowance(self.config,50,1),budget=1,scorer=scorer)
                saved=json.loads(out.read_text())
                self.assertGreater(saved['calls'],0)
                self.assertEqual(saved['status'],'running')
                self.assertEqual(len(saved['runs']),1)

    def test_upper_envelope_and_no_extrapolation(self):
        hull = frontier([(1,.5),(2,.4),(3,.9)],(4,1.))
        self.assertNotIn((2,.4), hull)
        self.assertAlmostEqual(performance_at_budget(hull,2),.7)
        self.assertIsNone(performance_at_budget(hull,5))
        self.assertAlmostEqual(auc(hull),2.6)
        self.assertIsNone(frontier([(0,1)],(0,1)))

    def test_metric_rejects_partial_and_simulated_runs(self):
        with self.assertRaises(ValueError):
            measurement({'status':'limit_reached','evaluated':1,'requested':2})
        report = {'status':'completed','evaluated':1,'requested':1,'accuracy':1.,
                  'runs':[{'key':'a','report':{'mode':'offline_simulation','status':'completed',
                          'totals':{'cost_complete':True,'cost_usd':0.,'latency_seconds':1.,
                                    'input_tokens':2,'output_tokens':2,'calls':1}}}]}
        with self.assertRaises(ValueError):
            measurement(report)
        self.assertEqual(measurement(report,True)['tokens'],4)

    def test_paper_scoring_and_full_humaneval_function(self):
        scorer = Scorer('python:3.13-slim',protocol='paper')
        self.assertEqual(scorer.score('The answer is C',{'scorer':'mmlu','answers':['C']}),1.)
        self.assertEqual(scorer.score(r'\boxed{\frac12}',{'scorer':'math','answers':['0.5']}),1.)
        self.assertEqual(scorer.score(r'\boxed{x = \sqrt2}',{'scorer':'math','answers':[r'\sqrt{2}']}),1.)
        row = {'scorer':'humaneval','code_prompt':'def f():\n', 'test':'def check(f): pass', 'entry_point':'f'}
        with patch('agentbalance.adaptive.scoring.subprocess.run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = 'AGENTBALANCE_TESTS_PASSED'
            self.assertEqual(scorer.score('```python\ndef f():\n    return 42\n```',row),1.)
            payload = json.loads(run.call_args_list[0].kwargs['input'])
            self.assertEqual(payload['program'].count('def f'),1)

    def test_native_dataset_ordering_and_disjoint_humaneval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('dev','val'):
                (root/split).mkdir()
                (root/split/'a.csv').write_text(''.join(f'{split}{i},a,b,c,d,A\n' for i in range(5)))
            prepare_mmlu(root, root/'prepared', train_count=3,eval_count=2)
            self.assertEqual(len((root/'prepared/mmlu-train.jsonl').read_text().splitlines()),3)
            rows = [{'task_id':str(i),'prompt':f'def f{i}():\n','test':'test', 'entry_point':f'f{i}'} for i in range(4)]
            raw = root/'code.jsonl'
            raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            prepare_humaneval(raw, root/'prepared',train_count=2)
            train = [json.loads(r)['source_id'] for r in (root/'prepared/humaneval-train.jsonl').read_text().splitlines()]
            test = [json.loads(r)['source_id'] for r in (root/'prepared/humaneval-eval.jsonl').read_text().splitlines()]
            self.assertEqual(train,['0','1'])
            self.assertEqual(test,['2','3'])

    def test_optimizer_resume_matches_uninterrupted_updates(self):
        config = replace(self.config, ablation='no-gating')
        original = build_controller(config, TextEncoder('hash-test',16),hidden=16)
        full, partial = copy.deepcopy(original), copy.deepcopy(original)
        rows = [{'task':task,'key':task_key(task),'answers':['42']} for task in ['first','second']]
        with tempfile.TemporaryDirectory() as directory, patch('agentbalance.providers.complete',
                return_value=Completion('42',5,5,'provider','stop')):
            root=Path(directory)
            torch.manual_seed(100)
            train_policy(full,rows,root/'full.pt',RunAllowance(config,100,10),epochs=2,
                         budget=1, cost_weight=0,latency_weight=0,reward_units='raw')
            torch.manual_seed(100)
            result=train_policy(partial,rows,root/'partial.pt',RunAllowance(config,6,10),epochs=2,
                                budget=1, cost_weight=0,latency_weight=0,reward_units='raw')
            self.assertEqual(result['status'],'limit_reached')
            resumed=load_controller(root/'partial.pt',config,allow_untrained=True)
            train_policy(resumed,rows,root/'resumed.pt',RunAllowance(config,100,10),epochs=2,
                         budget=1, cost_weight=0,latency_weight=0,reward_units='raw',resume_state=root/'partial.optimizer.pt')
            for key, value in full.state_dict().items():
                self.assertTrue(torch.equal(value,resumed.state_dict()[key]),key)


if __name__ == '__main__':
    unittest.main()
