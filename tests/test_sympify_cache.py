# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests for TORCH_SPYRE_SYMPIFY_CACHE (memoized wrapper sympify).

Run with:
    python -m pytest tests/test_sympify_cache.py -v
"""

import inspect
import unittest

import sympy
from sympy.core.parameters import evaluate

import torch_spyre._inductor.config as spyre_config
from torch_spyre._inductor import op_spec, spyre_kernel, wrapper
from torch_spyre._inductor.op_spec import _cached_sympify, wrapper_sympify

# Expression strings as emitted by sympy_str() into generated wrapper modules.
_EXPRS = [
    "0",
    "1",
    "4096",
    "c0",
    "c4",
    "z0",
    "z0 + 7",
    "c1 + 40",
    "floor(c1/32)",
    "floor(c1/64) + 192",
    "Mod(c1, 32)",
    "Mod(core_id, 32)",
    "Mod(floor(core_id/8), 4)",
    "Mod(floor(Mod(core_id, 4)), 4)",
    "Mod(floor(Mod(floor(core_id/8), 4)), 4)",
    "_tile_adv_while_loop_body_graph_0_0_op7_lvl0",
    "floor(1024*_tile_adv_while_loop_body_graph_0_0_op7_lvl0)",
]

# A fragment of a generated OpSpec, evaluated the way a wrapper module is.
_GENERATED = (
    "({sympify('c0'): (sympify('2'), 1), sympify('c1'): (sympify('4096'), 32)}, "
    "{sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 32)')}, "
    "[sympify('floor(c1/32)'), sympify('c0'), sympify('Mod(c1, 32)')])"
)


def _signature(expr):
    """Structure plus the assumptions of every free symbol."""
    symbols = sorted(
        (s.name, tuple(sorted(s.assumptions0.items()))) for s in expr.free_symbols
    )
    return sympy.srepr(expr), symbols


class TestWrapperSympify(unittest.TestCase):
    def setUp(self):
        _cached_sympify.cache_clear()

    def tearDown(self):
        _cached_sympify.cache_clear()

    def test_on_equals_off_equals_sympy(self):
        with spyre_config.patch(sympify_cache=False):
            off = [wrapper_sympify(s) for s in _EXPRS]
        with spyre_config.patch(sympify_cache=True):
            on = [wrapper_sympify(s) for s in _EXPRS]
            on_again = [wrapper_sympify(s) for s in _EXPRS]
        for text, a, b, c in zip(_EXPRS, off, on, on_again):
            ref = sympy.sympify(text)
            for got in (a, b, c):
                self.assertEqual(got, ref, text)
                self.assertEqual(_signature(got), _signature(ref), text)

    def test_cache_hit_returns_equal_object(self):
        with spyre_config.patch(sympify_cache=True):
            first = wrapper_sympify("Mod(floor(core_id/8), 4)")
            hits = _cached_sympify.cache_info().hits
            second = wrapper_sympify("Mod(floor(core_id/8), 4)")
        self.assertEqual(_cached_sympify.cache_info().hits, hits + 1)
        self.assertEqual(second, first)
        self.assertIs(second, first)
        self.assertEqual(second, sympy.sympify("Mod(floor(core_id/8), 4)"))

    def test_non_str_passes_through(self):
        c0 = sympy.Symbol("c0")
        args = [4, 2.5, c0, sympy.Integer(7), sympy.floor(c0 / 32), [1, "c0"]]
        with spyre_config.patch(sympify_cache=True):
            for a in args:
                self.assertEqual(wrapper_sympify(a), sympy.sympify(a))
            # Extra sympify arguments also bypass the cache.
            got = wrapper_sympify("0.5*c0", rational=True)
            self.assertEqual(got, sympy.sympify("0.5*c0", rational=True))
        self.assertEqual(_cached_sympify.cache_info().currsize, 0)

    def test_off_never_caches(self):
        with spyre_config.patch(sympify_cache=False):
            for s in _EXPRS + _EXPRS:
                wrapper_sympify(s)
        info = _cached_sympify.cache_info()
        self.assertEqual((info.hits, info.misses, info.currsize), (0, 0, 0))

    def test_evaluate_flag_is_honoured(self):
        text = "2*3 + c0"
        with spyre_config.patch(sympify_cache=True):
            evaluated = wrapper_sympify(text)
            with evaluate(False):
                unevaluated = wrapper_sympify(text)
                ref = sympy.sympify(text)
        self.assertEqual(sympy.srepr(unevaluated), sympy.srepr(ref))
        self.assertEqual(evaluated, sympy.sympify(text))
        self.assertNotEqual(sympy.srepr(evaluated), sympy.srepr(unevaluated))

    def test_generated_fragment_on_equals_off(self):
        results = []
        for on in (False, True, True):
            ns = {}
            exec(
                "from torch_spyre._inductor.op_spec import wrapper_sympify as sympify",
                ns,
            )
            with spyre_config.patch(sympify_cache=on):
                results.append(eval(_GENERATED, ns))
        ref = eval(_GENERATED, {"sympify": sympy.sympify})
        for got in results:
            self.assertEqual(got, ref)

    def test_codegen_does_not_read_switch(self):
        # The generated code must be identical with the switch on and off.
        for module in (wrapper, spyre_kernel):
            self.assertNotIn("sympify_cache", inspect.getsource(module))
        header = inspect.getsource(wrapper.SpyrePythonWrapperCodegen.write_header)
        self.assertIn(
            "from torch_spyre._inductor.op_spec import wrapper_sympify as sympify",
            header,
        )
        self.assertIn("sympify_cache", inspect.getsource(op_spec.wrapper_sympify))


if __name__ == "__main__":
    unittest.main()
