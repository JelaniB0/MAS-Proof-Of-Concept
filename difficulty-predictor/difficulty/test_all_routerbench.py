"""All-family coverage, prompt extraction, grouping and masked evaluation invariants."""

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from .all_routerbench import assemble_main, partition_rows, family_name, normalize_condition_export
from .evaluation import paired_interval
from .inputs import task_text, input_text, input_version, INPUT_VERSION, TASK_INPUT_VERSION
from .safe_pickle import read_routerbench


class AllRouterBenchTests(unittest.TestCase):
    def test_per_model_training_reload_and_ordered_pool_aggregation(self):
        from .data import prepare
        from .model import DifficultyPredictor,load_splits
        from .research import run_suite
        from .test_research import local_encoder
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            local_encoder(root/'encoder')
            frame=pd.DataFrame([{'sample_id':str(i),'eval_name':'math','prompt':f'question add {i}',
                                 'a':None if i%5==0 else i%2,'b':.5,'c':1,'d':0} for i in range(100)])
            frame.to_json(root/'source.jsonl',orient='records',lines=True)
            manifest=prepare(root/'source.jsonl',root/'data')
            order=list(reversed([m for g in ['weak','intermediate','strong'] for m in manifest['groups'][g]]))
            config=[{'name':'per_model','input_mode':'task','balanced':False,'loss':'mse','prediction_level':'model',
                     'model_order':order,'groups':manifest['groups']}]
            run_suite(root/'data',root/'encoder',root/'run',device='cpu',seeds=(42,),epochs=1,patience=1,
                      configurations=config,compare_legacy=False)
            predictor=DifficultyPredictor.load(root/'run/selected_model')
            features=predictor.features(['question add 1'])[0]
            self.assertEqual(set(features['model_difficulty']),set(order))
            for group in ['weak','intermediate','strong']:
                expected=np.mean([features['model_difficulty'][m] for m in manifest['groups'][group]])
                self.assertAlmostEqual(features['difficulty'][group],expected,places=6)
            report=json.loads((root/'run/candidates/per_model/seed_42/training.json').read_text())
            self.assertGreater(report['observed_training_targets'],3*len(load_splits(root/'data')['train']))

    def test_task_extraction_preserves_unanswered_question_and_options(self):
        footer='Print only a single choice  from "A" or "B" without explanation. Answer:'
        example="Please answer with the letter of the correct answer.\n\nOld problem?\nA) old\nB) other\n"+footer+"\nA\n\n"
        question="Please answer with the letter of the correct answer.\n\nNew problem?\nA) first\nB) second\n"+footer
        self.assertEqual(task_text(example+example+question),"New problem?\nA) first\nB) second")
        self.assertEqual(task_text("Explain your answer: A or B?"),"Explain your answer: A or B?")
        self.assertEqual(input_text(question,"task"),task_text(question))
        self.assertEqual(input_version("question"),INPUT_VERSION)
        self.assertEqual(input_version("task"),TASK_INPUT_VERSION)

    def test_all_sources_retained_and_unknown_math_pool_not_imputed(self):
        families=["grade-school-math","mbpp","arc-challenge","hellaswag","winogrande","mmlu-physics","mtbench","mtbench-reference","a-new-eval"]
        rows=[{"sample_id":f"{family}.{i}","eval_name":family,"prompt":f"{family} question {i}","a":0,"b":1,"c":.75}
              for family in families for i in range(30)]
        zero=pd.DataFrame(rows)
        five=zero.copy()
        five.loc[five.eval_name=='arc-challenge','a']=1
        five=pd.concat([five,pd.DataFrame([{"sample_id":"arc.extra","eval_name":"arc-challenge","prompt":"Unique extra ARC question", "a":1,"b":1,"c":1}])],ignore_index=True)
        groups={"weak":["a"],"intermediate":["b"],"strong":["c"]}
        gsm={r['sample_id']:{"a":None,"b":1,"c":0} for r in rows if r['eval_name']=='grade-school-math'}
        prepared,metrics=assemble_main(zero,five,groups,gsm)
        self.assertEqual(len(prepared),len(zero)+1)
        self.assertEqual({r['source_eval_name'] for r in prepared},set(families))
        math=next(r for r in prepared if r['eval_name']=='grade-school-math')
        self.assertIsNone(math['difficulty']['weak'])
        self.assertEqual(math['difficulty']['strong'],1)
        arc=next(r for r in prepared if r['eval_name']=='arc-challenge')
        self.assertEqual(arc['scores']['a'],1)
        self.assertIn("reference_checked_numeric_correctness",metrics['grade-school-math'])
        self.assertEqual(family_name('mtbench-math'),'mtbench')
        self.assertEqual(family_name('a-new-eval'),'additional-evals')

    def test_duplicate_tasks_and_rag_article_groups_never_cross_folds(self):
        rows=[{"sample_id":f"{i}.{j}","prompt":f"Question {j} about article {i}?","article_source":f"article-{i}"}
              for i in range(80) for j in range(3)]
        partition_rows(rows,by_article=True)
        for i in range(80):
            self.assertEqual(len({r['split'] for r in rows if r['article_source']==f'article-{i}'}),1)
        self.assertEqual(len({r['split'] for r in rows}),3)
        duplicate=[{"sample_id":str(i),"prompt":f"Task {i}"} for i in range(120)]
        duplicate.append({"sample_id":"alias","prompt":"Task   0"})
        partition_rows(duplicate)
        self.assertEqual(duplicate[0]['split'],duplicate[-1]['split'])

    def test_complementary_fewshot_rows_require_identical_actual_tasks(self):
        footer='Print only a single choice from "A" or "B" without explanation. Answer:'
        task='Actual question?\nA) first\nB) second\n'+footer
        rows=[{'sample_id':'arc.1','eval_name':'arc-challenge','prompt':'Example one\n'+footer+'\nA\n\n'+task,'a':1,'b':None,'c':None},
              {'sample_id':'arc.1','eval_name':'arc-challenge','prompt':'Example two\n'+footer+'\nB\n\n'+task,'a':None,'b':0,'c':1}]
        groups={'weak':['a'],'intermediate':['b'],'strong':['c']}
        merged=normalize_condition_export(pd.DataFrame(rows),groups)
        self.assertEqual(len(merged),1)
        self.assertEqual(merged[0]['scores'],{'a':1,'b':0,'c':1})
        self.assertEqual(merged[0]['benchmark_prompt_variants'],2)
        rows[1]['prompt']='A different task'
        with self.assertRaisesRegex(ValueError,'conflicting actual tasks'):
            normalize_condition_export(pd.DataFrame(rows),groups)

    def test_bootstrap_matches_pool_macro_estimand_with_missing_targets(self):
        truth=np.zeros((4,3),dtype=float)
        truth[1:,0]=np.nan
        reference=np.full((4,3),.5)
        reference[0,0]=1
        candidate=np.zeros((4,3))
        rows=[{"sample_id":str(i),"eval_name":"math","bootstrap_cluster":"a" if i<2 else "b"} for i in range(4)]
        result=paired_interval(reference,candidate,truth,rows,repetitions=100)
        self.assertAlmostEqual(result['mean_mse_reduction'],.5)
        self.assertIn('cluster',result['method'])

    def test_safe_reader_noncontiguous_blocks_and_bounded_inspection(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'data.pkl'
            frame=pd.DataFrame({'left':[1.,2.],'text':pd.Series(['x'*2000,'short'],dtype=object),'right':[3.,4.]})
            frame.columns=pd.Index(['left','text','right'],dtype=object)
            frame.to_pickle(p,protocol=5)
            full=read_routerbench(p)
            self.assertEqual(full.to_dict('list'),frame.to_dict('list'))
            self.assertEqual(full['left'].dtype,np.dtype('float64'))
            limited=read_routerbench(p,max_text_characters=512)
            self.assertTrue(limited.loc[0,'text'].startswith('__OMITTED_LONG_TEXT_SHA256__:'))
            self.assertEqual(limited.loc[1,'right'],4)
            self.assertEqual(limited.attrs['inspection_text_limit'],512)


if __name__=='__main__':
    unittest.main()
