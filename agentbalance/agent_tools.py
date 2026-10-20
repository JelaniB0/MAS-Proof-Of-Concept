"""Reference-style math execution and public-example code checks in Docker."""

import json
import re
import subprocess
from uuid import uuid4

HARNESS = r'''
import contextlib, io, json, sys
row = json.load(sys.stdin)
scope = {}
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec('from typing import *\n' + row['program'], scope)
        for test in row.get('tests', []):
            exec(test, scope)
    print(json.dumps({'passed': True, 'answer': str(scope.get('answer')), 'feedback': 'Public examples passed'}))
except BaseException as exc:
    print(json.dumps({'passed': False, 'answer': None, 'feedback': type(exc).__name__ + ': ' + str(exc)[:1000]}))
'''


def extract_code(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match[1] if match else text


def public_examples(task):
    # The reference extracts only visible >>> examples, never hidden grader tests.
    lines = [line.strip() for line in task.splitlines() if line.strip()]
    tests = []
    for index, line in enumerate(lines[:-1]):
        if line.startswith('>>>') and not lines[index+1].startswith(('>>>', '...')):
            tests.append(f"assert {line[3:].strip()} == {lines[index+1]}")
    return tests


def run_code(image, program, tests=(), timeout=10):
    if not image or image.startswith('-'):
        raise ValueError("Agent tools need a valid, locally available Python Docker image")
    name = 'agentbalance-tool-'+uuid4().hex
    command = ['docker','run','--rm','--pull','never','--name',name,'--network','none','--read-only',
               '--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','64',
               '--memory','256m','--cpus','1','--user','65534:65534',
               '--tmpfs','/tmp:rw,noexec,nosuid,size=16m','-i',image,'python','-I','-c',HARNESS]
    try:
        result = subprocess.run(command, input=json.dumps({'program':program,'tests':list(tests)}),
                                text=True, capture_output=True, timeout=timeout)
        if result.returncode in {125,126,127}:
            raise ValueError("Agent tool Docker container failed to start")
        if result.returncode:
            return {'passed':False,'answer':None,'feedback':'Code exited unsuccessfully'}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {'passed':False,'answer':None,'feedback':'Code execution timed out'}
    except json.JSONDecodeError:
        return {'passed':False,'answer':None,'feedback':'Code did not return a valid tool result'}
    finally:
        try:
            subprocess.run(['docker','rm','-f',name], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError('Could not confirm agent tool cleanup') from exc


def check_contributions(image, task, contributions):
    tests = public_examples(task)
    if not tests:
        return contributions, None, []
    updated, traces = dict(contributions), []
    for role, text in contributions.items():
        if not text.lstrip().startswith('```python'):
            continue
        result = run_code(image, extract_code(text), tests)
        traces.append({'role':role, 'public_tests':len(tests), **result})
        if result['passed']:
            return updated, text, traces
        updated[role] = 'Public-example execution feedback: '+result['feedback']+'\n'+text
    return updated, None, traces
