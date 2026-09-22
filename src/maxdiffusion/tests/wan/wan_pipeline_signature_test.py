"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Check that Wan pipeline calls use keywords accepted by local helpers.
"""

import ast
import os

import pytest

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pipelines", "wan")


def _module_paths():
  return sorted(
      os.path.join(PIPELINE_DIR, name) for name in os.listdir(PIPELINE_DIR) if name.endswith(".py") and name != "__init__.py"
  )


def _accepted_keywords(func: ast.FunctionDef):
  spec = func.args
  if spec.kwarg is not None:
    return None  # **kwargs accepts anything
  names = {a.arg for a in spec.args} | {a.arg for a in spec.kwonlyargs} | {a.arg for a in spec.posonlyargs}
  return names


@pytest.mark.parametrize("path", _module_paths(), ids=os.path.basename)
def test_pipeline_keyword_arguments_match_local_definitions(path):
  tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
  defs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
  problems = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
      continue
    target = defs.get(node.func.id)
    if target is None:
      continue
    accepted = _accepted_keywords(target)
    if accepted is None:
      continue
    for kw in node.keywords:
      if kw.arg is not None and kw.arg not in accepted:
        problems.append(f"{os.path.basename(path)}:{node.lineno}: {node.func.id}() does not accept '{kw.arg}'")
  assert not problems, "\n".join(problems)
