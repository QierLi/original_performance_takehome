"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        """Allocate a VLEN vector constant via broadcast from a scalar."""
        if val not in self.vconst_map:
            scalar = self.scratch_const(val)
            vec = self.alloc_scratch(name, length=VLEN)
            self.add("valu", ("vbroadcast", vec, scalar))
            self.vconst_map[val] = vec
        return self.vconst_map[val]

    def build_vhash(self, vval, vtmp1, vtmp2):
        """Vectorized 6-stage hash on VLEN elements in parallel."""
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vc1 = self.scratch_vconst(val1)
            vc3 = self.scratch_vconst(val3)
            self.add("valu", (op1, vtmp1, vval, vc1))
            self.add("valu", (op3, vtmp2, vval, vc3))
            self.add("valu", (op2, vval, vtmp1, vtmp2))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel: processes VLEN (8) walkers per iteration using
        vload/vstore/valu. Gather for tree node lookup uses load_offset.
        """
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_init = self.alloc_scratch("tmp_init")

        # Load header from memory
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_init, i))
            self.add("load", ("load", self.scratch[v], tmp_init))

        # Vector constants
        vzero = self.scratch_vconst(0, "vzero")
        vone = self.scratch_vconst(1, "vone")
        vtwo = self.scratch_vconst(2, "vtwo")

        # Broadcast scalar params to vectors
        vn_nodes = self.alloc_scratch("vn_nodes", VLEN)
        self.add("valu", ("vbroadcast", vn_nodes, self.scratch["n_nodes"]))
        vforest_p = self.alloc_scratch("vforest_p", VLEN)
        self.add("valu", ("vbroadcast", vforest_p, self.scratch["forest_values_p"]))

        self.add("flow", ("pause",))

        # Vector scratch registers
        vidx = self.alloc_scratch("vidx", VLEN)
        vval = self.alloc_scratch("vval", VLEN)
        vnode_val = self.alloc_scratch("vnode_val", VLEN)
        vtmp1 = self.alloc_scratch("vtmp1", VLEN)
        vtmp2 = self.alloc_scratch("vtmp2", VLEN)
        vtmp3 = self.alloc_scratch("vtmp3", VLEN)
        vaddr = self.alloc_scratch("vaddr", VLEN)

        for round_idx in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)

                # --- Load 8 indices via vload (contiguous) ---
                self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                self.add("load", ("vload", vidx, tmp_addr))

                # --- Load 8 values via vload (contiguous) ---
                self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                self.add("load", ("vload", vval, tmp_addr))

                # --- Gather: node_val[lane] = mem[forest_values_p + idx[lane]] ---
                self.add("valu", ("+", vaddr, vforest_p, vidx))
                for lane in range(0, VLEN, 2):
                    self.instrs.append({
                        "load": [
                            ("load_offset", vnode_val, vaddr, lane),
                            ("load_offset", vnode_val, vaddr, lane + 1),
                        ]
                    })

                # --- val = myhash(val ^ node_val) ---
                self.add("valu", ("^", vval, vval, vnode_val))
                self.build_vhash(vval, vtmp1, vtmp2)

                # --- idx = 2*idx + (1 if val%2==0 else 2) ---
                self.add("valu", ("&", vtmp1, vval, vone))  # &= instead of % -> = 
                self.add("flow", ("vselect", vtmp3, vtmp1, vtwo, vone))
                self.add("valu", ("*", vidx, vidx, vtwo))
                self.add("valu", ("+", vidx, vidx, vtmp3))

                # --- idx = 0 if idx >= n_nodes else idx ---
                self.add("valu", ("<", vtmp1, vidx, vn_nodes))
                self.add("flow", ("vselect", vidx, vtmp1, vidx, vzero))

                # --- Store back 8 indices and 8 values ---
                self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                self.add("store", ("vstore", tmp_addr, vidx))
                self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                self.add("store", ("vstore", tmp_addr, vval))

        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
