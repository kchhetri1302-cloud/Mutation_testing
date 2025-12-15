#!/usr/bin/env python3
"""
Mutation Generation Benchmark
=============================

Compares ONLY mutation generation (not test execution) across tools:
- Your Tool
- Mutmut
- Cosmic Ray  
- MutPy

Metrics:
- Number of mutants generated
- Generation time
- Validity rate (syntactically correct mutants)
- Mutation operator coverage

Usage:
    python benchmark_generation.py -p programs/Problem1.py
    python benchmark_generation.py -p programs/ --all
    python benchmark_generation.py --check

Requirements:
    pip install mutmut cosmic-ray mutpy parso
"""

import subprocess
import json
import time
import ast
import sys
import os
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import csv


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MutantInfo:
    """Information about a single mutant."""
    id: str
    operator: str
    location: Tuple[int, int]  # (line, column)
    is_valid: bool
    original_snippet: str = ""
    mutated_snippet: str = ""


@dataclass 
class GenerationResult:
    """Result of mutation generation from one tool."""
    tool: str
    program: str
    total_mutants: int
    valid_mutants: int
    invalid_mutants: int
    validity_rate: float
    generation_time_seconds: float
    operators_used: List[str]
    mutants: List[MutantInfo] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ComparisonResult:
    """Comparison across all tools for one program."""
    program: str
    lines_of_code: int
    results: Dict[str, GenerationResult]
    unique_to_tool: Dict[str, int]  # Mutants unique to each tool


# ============================================================================
# TOOL GENERATORS
# ============================================================================

class MutationGenerator:
    """Base class for mutation generators."""
    
    def is_available(self) -> bool:
        raise NotImplementedError
    
    def get_version(self) -> str:
        raise NotImplementedError
    
    def generate(self, program: Path) -> GenerationResult:
        raise NotImplementedError
    
    def _check_validity(self, code: str) -> bool:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class YourToolGenerator(MutationGenerator):
    """Generator using your mutation testing tool."""
    
    def __init__(self, tool_path: Path, behavior_only: bool = True, 
                 higher_order_k: int = 1, max_per_k: int = 1000):
        self.tool_path = tool_path
        self.behavior_only = behavior_only
        self.higher_order_k = higher_order_k  # 1 = first-order, 2+ = higher-order
        self.max_per_k = max_per_k
        self._add_src_to_path()
    
    def _add_src_to_path(self):
        """Add src directory to Python path."""
        # Try multiple possible src locations
        possible_src_dirs = [
            self.tool_path.parent / "src",
            Path.cwd() / "src",
            Path(__file__).parent / "src",
            Path(__file__).parent.parent / "src",
        ]
        
        for src_dir in possible_src_dirs:
            if src_dir.exists() and str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
                break
    
    def is_available(self) -> bool:
        try:
            # Try to import mutation_engine to check availability
            from mutation_engine import MutationEngine
            return True
        except ImportError:
            return False
    
    def get_version(self) -> str:
        try:
            from mutation_engine import __version__
            return __version__
        except (ImportError, AttributeError):
            return "1.0.0"
    
    def generate(self, program: Path) -> GenerationResult:
        start_time = time.time()
        
        try:
            # Import after adding src to path
            from mutation_engine import MutationEngine, MutationCombiner
            
            source_code = program.read_text()
            engine = MutationEngine()
            
            # Select mutations based on mode
            if self.behavior_only:
                # Use behavior-changing mutations
                BEHAVIOR_CHANGING = [
                    13, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
                    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 
                    52, 53, 54, 55
                ]
                mutations_to_use = BEHAVIOR_CHANGING
            else:
                mutations_to_use = None  # All mutations
            
            mutants = []
            operators_used = set()
            valid_count = 0
            
            if self.higher_order_k > 1:
                # Higher-order mutation mode
                combiner = MutationCombiner(engine)
                
                # Generate mutants for each k level
                for k in range(1, self.higher_order_k + 1):
                    # Use generate_combinations_from_mutants for better mutation coverage
                    k_results = combiner.generate_combinations_from_mutants(
                        code=source_code,
                        k=k,
                        mutations_to_use=mutations_to_use,
                        max_combinations=self.max_per_k,
                        line_range=None
                    )
                    
                    for r in k_results:
                        is_valid = r.is_valid and self._check_validity(r.mutated_code)
                        if is_valid:
                            valid_count += 1
                        
                        # Get operator name(s)
                        if hasattr(r, 'mutation_names') and r.mutation_names:
                            operator_name = "+".join(r.mutation_names)
                        elif hasattr(r, 'pattern_name'):
                            operator_name = r.pattern_name
                        else:
                            operator_name = f"k{k}_combined"
                        
                        # Get location
                        location = (0, 0)
                        if hasattr(r, 'locations') and r.locations:
                            # Use first location
                            loc = r.locations[0]
                            location = (loc.line, loc.column)
                        elif hasattr(r, 'location') and r.location:
                            location = (r.location.line, r.location.column)
                        
                        mutants.append(MutantInfo(
                            id=f"k{k}_{len(mutants)}",
                            operator=operator_name,
                            location=location,
                            is_valid=is_valid,
                        ))
                        
                        # Track operators
                        if hasattr(r, 'mutation_names') and r.mutation_names:
                            for name in r.mutation_names:
                                operators_used.add(name)
                        elif hasattr(r, 'pattern_name'):
                            operators_used.add(r.pattern_name)
                        else:
                            operators_used.add(operator_name)
            else:
                # First-order mutation mode (k=1)
                results = engine.mutate(source_code, mutations=mutations_to_use, behavior_only=self.behavior_only)
                
                for r in results:
                    is_valid = r.is_valid and self._check_validity(r.mutated_code)
                    if is_valid:
                        valid_count += 1
                    
                    loc = (r.location.line, r.location.column) if r.location else (0, 0)
                    
                    mutants.append(MutantInfo(
                        id=r.mutation_id or f"mutant_{len(mutants)}",
                        operator=r.pattern_name,
                        location=loc,
                        is_valid=is_valid,
                    ))
                    operators_used.add(r.pattern_name)
            
            elapsed = time.time() - start_time
            
            # Build tool name based on configuration
            tool_name = "YourTool"
            if self.behavior_only:
                tool_name += "-behavior"
            if self.higher_order_k > 1:
                tool_name += f"-k{self.higher_order_k}"
            
            return GenerationResult(
                tool=tool_name,
                program=program.name,
                total_mutants=len(mutants),
                valid_mutants=valid_count,
                invalid_mutants=len(mutants) - valid_count,
                validity_rate=valid_count / len(mutants) if mutants else 0,
                generation_time_seconds=elapsed,
                operators_used=sorted(operators_used),
                mutants=mutants,
                success=True
            )
            
        except Exception as e:
            import traceback
            return GenerationResult(
                tool="YourTool",
                program=program.name,
                total_mutants=0, valid_mutants=0, invalid_mutants=0,
                validity_rate=0, generation_time_seconds=time.time() - start_time,
                operators_used=[], mutants=[],
                success=False, error_message=f"{str(e)}\n{traceback.format_exc()}"
            )

class MutmutGenerator(MutationGenerator):
    """Generator using Mutmut (supports v2.x and v3.x)."""
    
    def is_available(self) -> bool:
        try:
            result = subprocess.run(["mutmut", "--version"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def get_version(self) -> str:
        try:
            result = subprocess.run(["mutmut", "--version"], capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _is_v3(self) -> bool:
        """Check if mutmut is version 3.x (different CLI)."""
        version = self.get_version()
        return "version 3" in version or version.startswith("3.")
    
    def generate(self, program: Path) -> GenerationResult:
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Copy program
            tmp_program = tmpdir / program.name
            shutil.copy(program, tmp_program)
            
            # Make temp dir a package
            (tmpdir / "__init__.py").write_text("")
            
            # Prepare test file
            real_test_file = program.parent / f"test_{program.stem}.py"
            tmp_test_file = tmpdir / f"test_{program.stem}.py"
            
            if real_test_file.exists():
                shutil.copy(real_test_file, tmp_test_file)
            else:
                # Dummy test so Mutmut can run
                tmp_test_file.write_text(f"""
import unittest
import {program.stem}

class TestDummy(unittest.TestCase):
    def test_pass(self):
        pass

if __name__ == '__main__':
    unittest.main()
""")
            
            try:
                if self._is_v3():
                    mutants = self._generate_v3(tmp_program, tmpdir)
                else:
                    mutants = self._generate_v2(tmp_program, tmpdir)
                
                elapsed = time.time() - start_time
                valid_count = len(mutants)
                operators = set(m.operator for m in mutants)
                
                return GenerationResult(
                    tool="Mutmut",
                    program=program.name,
                    total_mutants=len(mutants),
                    valid_mutants=valid_count,
                    invalid_mutants=0,
                    validity_rate=1.0 if mutants else 0,
                    generation_time_seconds=elapsed,
                    operators_used=sorted(operators),
                    mutants=mutants,
                    success=True
                )
                
            except subprocess.TimeoutExpired:
                return GenerationResult(
                    tool="Mutmut",
                    program=program.name,
                    total_mutants=0, valid_mutants=0, invalid_mutants=0,
                    validity_rate=0, generation_time_seconds=60,
                    operators_used=[], mutants=[],
                    success=False, error_message="Timeout"
                )
            except Exception as e:
                import traceback
                return GenerationResult(
                    tool="Mutmut",
                    program=program.name,
                    total_mutants=0, valid_mutants=0, invalid_mutants=0,
                    validity_rate=0, generation_time_seconds=time.time() - start_time,
                    operators_used=[], mutants=[],
                    success=False, error_message=f"{str(e)}\n{traceback.format_exc()}"
                )
    
    def _generate_v3(self, program: Path, work_dir: Path) -> List[MutantInfo]:
        """Generate mutations using Mutmut v3.x API (config file based)."""
        # v3 requires setup.cfg for configuration
        setup_cfg = work_dir / "setup.cfg"
        setup_cfg.write_text(f"""[mutmut]
paths_to_mutate={program.name}
tests_dir=.
""")
        
        # Clear any cache
        cache_dir = work_dir / ".mutmut-cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        
        # Run mutmut v3 (config-based, no CLI args for paths)
        result = subprocess.run(
            ["mutmut", "run"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=work_dir
        )
        
        # Parse output for mutant count
        # v3 output format: "34/34  🎉 23 🫥 0  ⏰ 0  🤔 0  🙁 11  🔇 0"
        mutants = []
        import re
        
        match = re.search(r'(\d+)/(\d+)\s+🎉', result.stdout)
        if match:
            total = int(match.group(2))
            for i in range(total):
                mutants.append(MutantInfo(
                    id=f"mutmut_{i+1}",
                    operator="mutmut_mutation",
                    location=(0, 0),
                    is_valid=True,
                ))
        
        return mutants
    
    def _generate_v2(self, program: Path, work_dir: Path) -> List[MutantInfo]:
        """Generate mutations using Mutmut v2.x API (CLI args)."""
        cache_dir = work_dir / ".mutmut-cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        
        cmd = [
            "mutmut", "run",
            "--paths-to-mutate", str(program),
            "--tests-dir", str(work_dir),
            "--no-progress",
            "--CI"
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=work_dir)
        
        show_cmd = ["mutmut", "show", "all"]
        show_result = subprocess.run(show_cmd, capture_output=True, text=True, cwd=work_dir)
        return self._parse_mutmut_output(show_result.stdout)
    
    def _parse_mutmut_output(self, output: str) -> List[MutantInfo]:
        mutants = []
        for line in output.splitlines():
            if ":" in line and not line.startswith("#"):
                mutant_id, desc = line.split(":", 1)
                operator = self._classify_mutation(desc)
                mutants.append(MutantInfo(
                    id=mutant_id.strip(),
                    operator=operator,
                    location=(0, 0),
                    is_valid=True
                ))
        return mutants

    def _classify_mutation(self, description: str) -> str:
        desc = description.lower()
        if any(op in desc for op in ["+", "-", "*", "/"]):
            return "arithmetic_op"
        elif "==" in desc or "!=" in desc:
            return "comparison_op"
        elif "and" in desc or "or" in desc:
            return "logical_op"
        elif "true" in desc or "false" in desc:
            return "boolean_literal"
        elif "none" in desc:
            return "none_mutation"
        elif "return" in desc:
            return "return_mutation"
        else:
            return "other"


class MutPyGenerator(MutationGenerator):
    """Generator using MutPy."""
    
    def is_available(self) -> bool:
        try:
            # Try multiple approaches
            try:
                result = subprocess.run(["mutpy", "--version"], capture_output=True)
                if result.returncode == 0:
                    return True
            except FileNotFoundError:
                pass
            
            result = subprocess.run(
                [sys.executable, "-m", "mutpy", "--version"],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False
    
    def get_version(self) -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mutpy", "--version"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def generate(self, program: Path) -> GenerationResult:
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Copy program
            tmp_program = tmpdir / program.name
            shutil.copy(program, tmp_program)
            
            # Create __init__.py for module
            (tmpdir / "__init__.py").write_text("")
            
            # Create dummy test
            test_file = tmpdir / f"test_{program.stem}.py"
            test_file.write_text(f"""
import unittest
class TestDummy(unittest.TestCase):
    def test_pass(self):
        pass
if __name__ == '__main__':
    unittest.main()
""")
            
            try:
                # Run MutPy with --show-mutants to list mutations
                cmd = [
                    sys.executable, "-m", "mutpy",
                    "--target", program.stem,
                    "--unit-test", f"test_{program.stem}",
                    "--show-mutants",
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=tmpdir
                )
                
                elapsed = time.time() - start_time
                
                mutants = self._parse_mutpy_output(result.stdout, tmp_program)
                
                valid_count = sum(1 for m in mutants if m.is_valid)
                operators = set(m.operator for m in mutants)
                
                return GenerationResult(
                    tool="MutPy",
                    program=program.name,
                    total_mutants=len(mutants),
                    valid_mutants=valid_count,
                    invalid_mutants=len(mutants) - valid_count,
                    validity_rate=valid_count / len(mutants) if mutants else 0,
                    generation_time_seconds=elapsed,
                    operators_used=sorted(operators),
                    mutants=mutants,
                    success=True
                )
                
            except subprocess.TimeoutExpired:
                return GenerationResult(
                    tool="MutPy",
                    program=program.name,
                    total_mutants=0, valid_mutants=0, invalid_mutants=0,
                    validity_rate=0, generation_time_seconds=120,
                    operators_used=[], mutants=[],
                    success=False, error_message="Timeout"
                )
            except Exception as e:
                return GenerationResult(
                    tool="MutPy",
                    program=program.name,
                    total_mutants=0, valid_mutants=0, invalid_mutants=0,
                    validity_rate=0, generation_time_seconds=time.time() - start_time,
                    operators_used=[], mutants=[],
                    success=False, error_message=str(e)
                )
    
    def _parse_mutpy_output(self, output: str, program: Path) -> List[MutantInfo]:
        """Parse MutPy output to extract mutants."""
        mutants = []
        
        # MutPy output format:
        # [*] Mutant: <operator> at <location>
        #     <original code>
        #     <mutated code>
        
        import re
        
        # Pattern for mutant header
        mutant_pattern = re.compile(r'\[\*\]\s+Mutant:\s+(\w+)')
        location_pattern = re.compile(r'line\s+(\d+)')
        
        lines = output.split('\n')
        current_operator = None
        mutant_count = 0
        
        for i, line in enumerate(lines):
            # Check for mutant header
            match = mutant_pattern.search(line)
            if match:
                current_operator = match.group(1)
                
                # Try to find line number
                loc_match = location_pattern.search(line)
                line_num = int(loc_match.group(1)) if loc_match else 0
                
                mutant_count += 1
                mutants.append(MutantInfo(
                    id=f"mutpy_{mutant_count}",
                    operator=current_operator,
                    location=(line_num, 0),
                    is_valid=True,
                ))
        
        # If we didn't find mutants in output, try counting from summary
        if not mutants:
            count_match = re.search(r'(\d+)\s+mutants?\s+generated', output)
            if count_match:
                count = int(count_match.group(1))
                for i in range(count):
                    mutants.append(MutantInfo(
                        id=f"mutpy_{i+1}",
                        operator="unknown",
                        location=(0, 0),
                        is_valid=True,
                    ))
        
        return mutants


class CosmicRayGenerator(MutationGenerator):
    """Generator using Cosmic Ray."""
    
    def is_available(self) -> bool:
        try:
            result = subprocess.run(["cosmic-ray", "--version"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def get_version(self) -> str:
        try:
            result = subprocess.run(["cosmic-ray", "--version"], capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def generate(self, program: Path) -> GenerationResult:
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Copy program and make package
            tmp_program = tmpdir / program.name
            shutil.copy(program, tmp_program)
            (tmpdir / "__init__.py").write_text("")
            
            # Dummy test file
            test_file = tmpdir / f"test_{program.stem}.py"
            test_file.write_text(f"""
import unittest
import {program.stem}

class TestDummy(unittest.TestCase):
    def test_pass(self):
        pass
""")
            
            # CosmicRay config - module-path needs .py extension
            config_file = tmpdir / "cosmic-ray.toml"
            config_file.write_text(f"""
[cosmic-ray]
module-path = "{program.name}"
timeout = 10.0
excluded-modules = []
test-command = "python -m pytest test_{program.stem}.py -x"

[cosmic-ray.distributor]
name = "local"
""")
            
            session_file = tmpdir / "session.sqlite"
            
            try:
                init_cmd = ["cosmic-ray", "init", str(config_file), str(session_file)]
                subprocess.run(init_cmd, capture_output=True, text=True, timeout=60, cwd=tmpdir)
                
                elapsed = time.time() - start_time
                mutants = self._get_mutations_from_session(session_file)
                
                valid_count = len(mutants)
                operators = set(m.operator for m in mutants)
                
                return GenerationResult(
                    tool="CosmicRay",
                    program=program.name,
                    total_mutants=len(mutants),
                    valid_mutants=valid_count,
                    invalid_mutants=0,
                    validity_rate=1.0,
                    generation_time_seconds=elapsed,
                    operators_used=sorted(operators),
                    mutants=mutants,
                    success=True
                )
                
            except subprocess.TimeoutExpired:
                return GenerationResult(
                    tool="CosmicRay",
                    program=program.name,
                    total_mutants=0, valid_mutants=0, invalid_mutants=0,
                    validity_rate=0, generation_time_seconds=60,
                    operators_used=[], mutants=[],
                    success=False, error_message="Timeout"
                )
            except Exception as e:
                return GenerationResult(
                    tool="CosmicRay",
                    program=program.name,
                    total_mutants=0, valid_mutants=0, invalid_mutants=0,
                    validity_rate=0, generation_time_seconds=time.time() - start_time,
                    operators_used=[], mutants=[],
                    success=False, error_message=str(e)
                )
    
    def _get_mutations_from_session(self, session_file: Path) -> List[MutantInfo]:
        mutants = []
        try:
            import sqlite3
            conn = sqlite3.connect(session_file)
            cursor = conn.cursor()
            cursor.execute("SELECT job_id, operator_name FROM mutation_specs")
            for row in cursor.fetchall():
                job_id, operator = row
                mutants.append(MutantInfo(
                    id=f"cr_{job_id}",
                    operator=operator,
                    location=(0, 0),
                    is_valid=True
                ))
            conn.close()
        except Exception:
            pass
        return mutants


# ============================================================================
# BENCHMARK ORCHESTRATOR
# ============================================================================

class GenerationBenchmark:
    """Main benchmarking class for mutation generation."""
    
    def __init__(self, your_tool_path: Path, output_dir: Path,
                 behavior_only: bool = True, 
                 higher_order_k: int = 1,
                 max_per_k: int = 1000):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build generators dict with multiple YourTool configurations
        self.generators = {}
        
        # YourTool - First-order, behavior-only (default, most comparable to others)
        self.generators["YourTool-behavior"] = YourToolGenerator(
            your_tool_path, 
            behavior_only=True, 
            higher_order_k=1
        )
        
        # YourTool - First-order, all mutations
        self.generators["YourTool-all"] = YourToolGenerator(
            your_tool_path, 
            behavior_only=False, 
            higher_order_k=1
        )
        
        # YourTool - Higher-order if requested
        if higher_order_k > 1:
            self.generators[f"YourTool-behavior-k{higher_order_k}"] = YourToolGenerator(
                your_tool_path,
                behavior_only=True,
                higher_order_k=higher_order_k,
                max_per_k=max_per_k
            )
            self.generators[f"YourTool-all-k{higher_order_k}"] = YourToolGenerator(
                your_tool_path,
                behavior_only=False,
                higher_order_k=higher_order_k,
                max_per_k=max_per_k
            )
        
        # Other tools
        self.generators["Mutmut"] = MutmutGenerator()
        self.generators["MutPy"] = MutPyGenerator()
        self.generators["CosmicRay"] = CosmicRayGenerator()
        
        self.results: List[GenerationResult] = []
    
    def check_tools(self) -> Dict[str, bool]:
        """Check which tools are available."""
        print("\nChecking available tools...")
        available = {}
        for name, gen in self.generators.items():
            available[name] = gen.is_available()
            status = "✓" if available[name] else "✗"
            version = gen.get_version() if available[name] else "not installed"
            print(f"  {status} {name}: {version}")
        return available
    
    def benchmark_single(self, program: Path, tools: List[str] = None) -> ComparisonResult:
        """Benchmark mutation generation on a single program."""
        if tools is None:
            tools = [name for name, gen in self.generators.items() if gen.is_available()]
        
        source = program.read_text()
        loc = len(source.splitlines())
        
        print(f"\n  Program: {program.name} ({loc} lines)")
        print(f"  {'-'*50}")
        
        results = {}
        all_mutations = {}  # tool -> set of (operator, line) tuples
        
        for tool_name in tools:
            if tool_name not in self.generators:
                continue
            
            gen = self.generators[tool_name]
            if not gen.is_available():
                continue
            
            print(f"  {tool_name}...", end=" ", flush=True)
            result = gen.generate(program)
            results[tool_name] = result
            self.results.append(result)
            
            if result.success:
                print(f"{result.total_mutants} mutants, "
                      f"{result.valid_mutants} valid ({result.validity_rate*100:.0f}%), "
                      f"{result.generation_time_seconds:.2f}s")
                
                # Track mutations for uniqueness analysis
                all_mutations[tool_name] = set(
                    (m.operator, m.location[0]) for m in result.mutants
                )
            else:
                print(f"FAILED: {result.error_message[:80]}...")
                all_mutations[tool_name] = set()
        
        # Calculate unique mutations per tool
        unique_to_tool = {}
        for tool_name, mutations in all_mutations.items():
            other_mutations = set()
            for other_name, other_muts in all_mutations.items():
                if other_name != tool_name:
                    other_mutations.update(other_muts)
            unique_to_tool[tool_name] = len(mutations - other_mutations)
        
        return ComparisonResult(
            program=program.name,
            lines_of_code=loc,
            results=results,
            unique_to_tool=unique_to_tool
        )
    
    def benchmark_all(self, programs_dir: Path, pattern: str = "*.py",
                      tools: List[str] = None) -> List[ComparisonResult]:
        """Benchmark all programs in a directory."""
        programs = sorted(programs_dir.glob(pattern))
        
        # Filter out test files and __init__.py
        programs = [p for p in programs 
                   if not p.name.startswith('test_') 
                   and p.name != '__init__.py']
        
        if not programs:
            print(f"No programs found in {programs_dir}")
            return []
        
        print(f"\n{'='*60}")
        print(f"MUTATION GENERATION BENCHMARK")
        print(f"{'='*60}")
        print(f"Programs: {len(programs)}")
        
        comparisons = []
        for i, program in enumerate(programs, 1):
            print(f"\n[{i}/{len(programs)}] {program.name}")
            comparison = self.benchmark_single(program, tools)
            comparisons.append(comparison)
        
        return comparisons
    
    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        
        # Group by tool
        by_tool = {}
        for r in self.results:
            if r.tool not in by_tool:
                by_tool[r.tool] = []
            by_tool[r.tool].append(r)
        
        # Per-tool summary
        print(f"\n{'Tool':<15} {'Programs':<10} {'Total Muts':<12} {'Avg Valid%':<12} {'Avg Time':<10} {'Operators'}")
        print("-" * 80)
        
        for tool, results in sorted(by_tool.items()):
            successful = [r for r in results if r.success]
            total_mutants = sum(r.total_mutants for r in successful)
            avg_validity = sum(r.validity_rate for r in successful) / len(successful) if successful else 0
            avg_time = sum(r.generation_time_seconds for r in successful) / len(successful) if successful else 0
            
            # Count unique operators
            all_operators = set()
            for r in successful:
                all_operators.update(r.operators_used)
            
            print(f"{tool:<15} {len(successful):<10} {total_mutants:<12} "
                  f"{avg_validity*100:>6.1f}%     {avg_time:>6.2f}s    {len(all_operators)}")
        
        # Operator comparison
        print(f"\n{'-'*70}")
        print("MUTATION OPERATORS BY TOOL")
        print(f"{'-'*70}")
        
        for tool, results in sorted(by_tool.items()):
            successful = [r for r in results if r.success]
            all_operators = set()
            for r in successful:
                all_operators.update(r.operators_used)
            
            print(f"\n{tool} ({len(all_operators)} operators):")
            for op in sorted(all_operators)[:15]:  # Show first 15
                print(f"  • {op}")
            if len(all_operators) > 15:
                print(f"  ... and {len(all_operators) - 15} more")
    
    def save_results(self, filename: str = "generation_benchmark.json"):
        """Save results to JSON."""
        filepath = self.output_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(r) for r in self.results]
        }
        
        # Convert MutantInfo to dict
        for r in data["results"]:
            r["mutants"] = [asdict(m) if hasattr(m, '__dict__') else m for m in r.get("mutants", [])]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
    
    def save_csv(self, filename: str = "generation_benchmark.csv"):
        """Save summary to CSV."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Tool", "Program", "LOC", "Total Mutants", "Valid", "Invalid",
                "Validity%", "Time(s)", "Operators", "Success"
            ])
            
            for r in self.results:
                writer.writerow([
                    r.tool, r.program, 0,  # LOC would need to be tracked separately
                    r.total_mutants, r.valid_mutants, r.invalid_mutants,
                    f"{r.validity_rate*100:.1f}",
                    f"{r.generation_time_seconds:.3f}",
                    len(r.operators_used),
                    r.success
                ])
        
        print(f"CSV saved to: {filepath}")
    
    def generate_report(self, comparisons: List[ComparisonResult]):
        """Generate detailed comparison report."""
        report_file = self.output_dir / "benchmark_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Mutation Generation Benchmark Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Tool | Programs | Total Mutants | Avg Validity | Avg Time |\n")
            f.write("|------|----------|---------------|--------------|----------|\n")
            
            by_tool = {}
            for r in self.results:
                if r.tool not in by_tool:
                    by_tool[r.tool] = []
                by_tool[r.tool].append(r)
            
            for tool, results in sorted(by_tool.items()):
                successful = [r for r in results if r.success]
                total = sum(r.total_mutants for r in successful)
                avg_val = sum(r.validity_rate for r in successful) / len(successful) if successful else 0
                avg_time = sum(r.generation_time_seconds for r in successful) / len(successful) if successful else 0
                f.write(f"| {tool} | {len(successful)} | {total} | {avg_val*100:.1f}% | {avg_time:.2f}s |\n")
            
            # Per-program comparison
            f.write("\n## Per-Program Results\n\n")
            
            for comp in comparisons:
                f.write(f"### {comp.program}\n\n")
                f.write(f"Lines of code: {comp.lines_of_code}\n\n")
                
                f.write("| Tool | Mutants | Valid | Time |\n")
                f.write("|------|---------|-------|------|\n")
                
                for tool, result in comp.results.items():
                    if result.success:
                        f.write(f"| {tool} | {result.total_mutants} | {result.valid_mutants} | {result.generation_time_seconds:.2f}s |\n")
                    else:
                        f.write(f"| {tool} | FAILED | - | - |\n")
                
                f.write("\n")
        
        print(f"Report saved to: {report_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark mutation generation across tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_generation.py --check
  python benchmark_generation.py -p programs/Problem1.py
  python benchmark_generation.py -p programs/ --all
  python benchmark_generation.py -p programs/ --tools YourTool-behavior MutPy
  
  # Test higher-order mutations (your tool's unique feature!)
  python benchmark_generation.py -p programs/ -k 2
  python benchmark_generation.py -p programs/ -k 3 --max-per-k 50
  
  # Compare your tool configurations
  python benchmark_generation.py -p programs/ --tools YourTool-behavior YourTool-all

Tool Configurations:
  YourTool-behavior     First-order, behavior-changing mutations only (33 operators)
  YourTool-all          First-order, all mutations (56 operators)
  YourTool-behavior-kN  Higher-order (k=N), behavior-changing only
  YourTool-all-kN       Higher-order (k=N), all mutations
  Mutmut                Mutmut v2.4.4 (25 operators)
  MutPy                 MutPy v0.6.1 (31 operators, has higher-order support)
  CosmicRay             Cosmic Ray v8.3.5 (31 operators)
        """
    )
    
    parser.add_argument("-p", "--programs", type=Path,
                        help="Program file or directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmark_results"),
                        help="Output directory")
    parser.add_argument("--tool-path", type=Path, default=Path("mutate_and_test.py"),
                        help="Path to your mutation tool")
    parser.add_argument("--tools", nargs="+",
                        help="Tools to benchmark (see list above)")
    parser.add_argument("-k", "--higher-order", type=int, default=1,
                        help="Higher-order mutation level (1=first-order, 2+=higher-order)")
    parser.add_argument("--max-per-k", type=int, default=1000,
                        help="Max mutants per k-level for higher-order")
    parser.add_argument("--check", action="store_true",
                        help="Check available tools")
    parser.add_argument("--pattern", default="*.py",
                        help="File pattern")
    
    args = parser.parse_args()
    
    benchmark = GenerationBenchmark(
        your_tool_path=args.tool_path,
        output_dir=args.output,
        higher_order_k=args.higher_order,
        max_per_k=args.max_per_k
    )
    
    available = benchmark.check_tools()
    
    if args.check:
        print("\n  Available configurations:")
        for name in benchmark.generators.keys():
            print(f"    • {name}")
        return 0
    
    if not args.programs:
        parser.print_help()
        return 1
    
    tools = args.tools
    if not tools:
        # Default: compare YourTool-behavior against other tools
        tools = [name for name, avail in available.items() if avail]
        # Limit to most relevant comparisons
        priority = ["YourTool-behavior", "Mutmut", "MutPy", "CosmicRay"]
        if args.higher_order > 1:
            priority.insert(1, f"YourTool-behavior-k{args.higher_order}")
        tools = [t for t in priority if t in tools]
    
    print(f"\nUsing tools: {', '.join(tools)}")
    
    if args.programs.is_file():
        benchmark.benchmark_single(args.programs, tools)
    else:
        comparisons = benchmark.benchmark_all(args.programs, args.pattern, tools)
        benchmark.generate_report(comparisons)
    
    benchmark.print_summary()
    benchmark.save_results()
    benchmark.save_csv()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())