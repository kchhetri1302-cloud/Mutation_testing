"""
Mutation Testing Runner
=======================

Executes mutations and compares outputs to determine mutation scores.

This module handles EXECUTION only:
- Run original program to get expected output
- Run each mutant and compare outputs
- Categorize results (killed, survived, timeout, error, equivalent)
- Generate reports

Input discovery is handled by input_discovery.py.
Mutation generation is handled by mutation_engine.py.
"""

import json
import time
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Import shared utilities - single source of truth
from common import Colors, SafeExecutor, DiffGenerator


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _normalize_output(out: Optional[str]) -> str:
    """
    Normalize output for robust comparison.
    """
    if out is None:
        return ""
    # Normalize line endings and strip trailing whitespace
    return out.strip().replace("\r\n", "\n").replace("\r", "\n")


def _outputs_equal(original: Optional[str], mutant: Optional[str]) -> bool:
    """
    Compare outputs with normalization.
    """
    return _normalize_output(original) == _normalize_output(mutant)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MutantResult:
    """Result of executing a single mutant."""
    id: str
    status: str  # killed, survived, runtime_error, timeout, invalid, equivalent
    pattern: Optional[str] = None
    location: Optional[Tuple[int, int]] = None
    locations_str: Optional[str] = None  # For combined mutations: "L3, L5, L7"
    change_description: Optional[str] = None  # e.g., "a + b → a - b"
    original_output: Optional[str] = None
    mutant_output: Optional[str] = None
    stderr: Optional[str] = None
    runtime_ms: int = 0
    error_message: Optional[str] = None
    killed_by: Optional[str] = None  # Which input pattern killed this mutant
    mutant_source: Optional[str] = None  # Mutant source code for diff display
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        loc_dict = None
        if self.location:
            loc_dict = {"line": self.location[0], "column": self.location[1]}
        
        result = {
            "id": self.id,
            "pattern": self.pattern,
            "location": loc_dict,
            "locations_str": self.locations_str,
            "change_description": self.change_description,
            "status": self.status,
            "error_message": self.error_message,
            "killed_by": self.killed_by,
            "runtime_ms": self.runtime_ms,
        }
        
        # Include output comparison for killed mutants
        if self.status == "killed" and self.original_output is not None:
            result["original_output"] = self.original_output[:500] if self.original_output else None
            result["mutant_output"] = self.mutant_output[:500] if self.mutant_output else None
        
        # Include stderr for runtime errors
        if self.status == "runtime_error" and self.stderr:
            result["stderr"] = self.stderr[:500]
        
        return result


@dataclass
class MutationReport:
    """Final mutation testing report."""
    total_mutants: int
    killed: int
    survived: int
    runtime_errors: int
    timeouts: int
    invalid: int
    equivalent: int
    mutation_score: float
    time_taken_seconds: float
    results: List[MutantResult]
    input_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_mutants": self.total_mutants,
                "killed": self.killed,
                "survived": self.survived,
                "runtime_errors": self.runtime_errors,
                "timeouts": self.timeouts,
                "invalid": self.invalid,
                "equivalent": self.equivalent,
                "mutation_score": round(self.mutation_score, 4),
                "mutation_score_percent": f"{self.mutation_score * 100:.1f}%",
                "time_taken_seconds": round(self.time_taken_seconds, 2),
                "input_patterns": self.input_patterns
            },
            "results": [r.to_dict() for r in self.results]
        }


# ============================================================================
# WORKER FUNCTION (for parallel execution)
# ============================================================================

def _execute_mutant_task(args: Tuple) -> MutantResult:
    """
    Worker function for parallel execution.
    Must be module-level for multiprocessing.
    """
    (mutant_id, mutant_source, original_output, stdin_input, 
     pattern, location, timeout, input_name) = args
    
    result = SafeExecutor.execute(mutant_source, stdin_input=stdin_input, timeout=timeout)
    
    if result["status"] == "timeout":
        status = "timeout"
        error_msg = "Execution timed out"
    elif result["status"] == "runtime_error":
        status = "runtime_error"
        error_msg = result["stderr"][:200] if result["stderr"] else None
    elif not _outputs_equal(result["stdout"], original_output):
        status = "killed"
        error_msg = None
    else:
        status = "survived"
        error_msg = None
    
    return MutantResult(
        id=mutant_id,
        status=status,
        pattern=pattern,
        location=location,
        original_output=original_output,
        mutant_output=result["stdout"],
        stderr=result["stderr"],
        runtime_ms=result["runtime_ms"],
        error_message=error_msg,
        killed_by=input_name if status in ("killed", "runtime_error", "timeout") else None
    )


# ============================================================================
# MUTATION RUNNER
# ============================================================================

class MutationRunner:
    """
    Executes mutation testing.
    
    Usage:
        runner = MutationRunner(source_code, timeout=3)
        runner.add_mutants(mutation_results)  # From MutationEngine
        
        # Single input mode
        report = runner.run("5\\n3\\n")
        
        # Multi-input mode (for thoroughness)
        report = runner.run_multi({"pattern1": "input1", "pattern2": "input2"})
        
        runner.print_report()
        runner.save_report("results.json")
    """
    
    def __init__(
        self,
        original_source: str,
        timeout: int = 3,
        max_workers: int = 4
    ):
        """
        Initialize the mutation runner.
        
        Args:
            original_source: Source code of original program
            timeout: Per-mutant timeout in seconds
            max_workers: Number of parallel workers
        """
        self.original_source = original_source
        self.timeout = timeout
        self.max_workers = max_workers
        
        # Storage for mutants
        self.mutants: List[Dict[str, Any]] = []
        
        # Results
        self.report: Optional[MutationReport] = None
    
    def add_mutants(self, mutation_results: List[Any]):
        """
        Add mutants from MutationEngine output.
        
        Args:
            mutation_results: List of MutationResult or CombinedMutationResult objects
        """
        for i, mr in enumerate(mutation_results):
            # Handle both MutationResult and CombinedMutationResult
            is_combined = hasattr(mr, 'mutation_names') and isinstance(getattr(mr, 'mutation_names', None), list)
            
            if is_combined:
                mutant_id = f"combined_{i+1:04d}"
                pattern = "+".join(mr.mutation_names)
                
                # Get locations from combined result
                location = None
                all_locations = []
                if hasattr(mr, 'locations') and mr.locations:
                    all_locations = [(loc.line, loc.column) for loc in mr.locations]
                    # Use first location as primary
                    if all_locations:
                        location = all_locations[0]
                
                # Store all locations as string for display
                locations_str = None
                if len(all_locations) > 1:
                    locations_str = ", ".join(f"L{loc[0]}" for loc in all_locations)
                
                change_description = None
            else:
                mutant_id = getattr(mr, 'mutation_id', None) or f"mutant_{len(self.mutants):04d}"
                pattern = mr.pattern_name
                location = None
                locations_str = None
                if hasattr(mr, 'location') and mr.location:
                    location = (mr.location.line, mr.location.column)
                
                # Get change description if available
                change_description = getattr(mr, 'change_description', None)
            
            self.mutants.append({
                "id": mutant_id,
                "source": mr.mutated_code,
                "pattern": pattern,
                "location": location,
                "locations_str": locations_str,
                "change_description": change_description,
                "is_valid": mr.is_valid,
            })
    
    def run(self, stdin_input: str = "", parallel: bool = True) -> MutationReport:
        """
        Run mutation testing with single input.
        
        Args:
            stdin_input: Input to provide to programs
            parallel: Use parallel execution
            
        Returns:
            MutationReport with results
        """
        return self._run_internal({"default": stdin_input}, parallel)
    
    def run_multi(self, inputs: Dict[str, str], parallel: bool = True) -> MutationReport:
        """
        Run mutation testing with multiple inputs.
        
        A mutant is KILLED if ANY input produces different output.
        A mutant SURVIVES only if ALL inputs produce same output.
        
        Args:
            inputs: Dict mapping input_name -> stdin_string
            parallel: Use parallel execution
            
        Returns:
            MutationReport with results
        """
        return self._run_internal(inputs, parallel)
    
    def _run_internal(self, inputs: Dict[str, str], parallel: bool) -> MutationReport:
        """Internal implementation for both single and multi-input modes."""
        start_time = time.time()
        
        input_count = len(inputs)
        is_multi = input_count > 1
        
        print(f"\n{Colors.BOLD}MUTATION TESTING{Colors.RESET}")
        print(f"Mutants: {len(self.mutants)}")
        print(f"Inputs: {input_count}")
        print(f"Timeout: {self.timeout}s")
        print()
        
        # Step 1: Get expected outputs for each input
        print(f"{Colors.CYAN}Running original program...{Colors.RESET}")
        expected_outputs = {}
        
        for input_name, stdin_input in inputs.items():
            result = SafeExecutor.execute(self.original_source, stdin_input=stdin_input, timeout=self.timeout)
            
            if result["status"] != "success":
                raise RuntimeError(f"Original program failed on input '{input_name}': {result['status']}")
            
            expected_outputs[input_name] = result["stdout"]
            
            if is_multi:
                preview = _normalize_output(result["stdout"])[:30].replace("\n", "\\n")
                print(f"  {input_name}: {preview}...")
        
        # Step 2: Execute mutants
        print(f"\n{Colors.CYAN}Testing mutants...{Colors.RESET}")
        
        results: List[MutantResult] = []
        kill_stats: Dict[str, int] = {}
        
        # Process invalid mutants first
        for m in self.mutants:
            if not m["is_valid"]:
                results.append(MutantResult(
                    id=m["id"],
                    status="invalid",
                    pattern=m["pattern"],
                    location=m["location"],
                    locations_str=m.get("locations_str"),
                    error_message="Invalid mutation"
                ))
        
        valid_mutants = [m for m in self.mutants if m["is_valid"]]
        
        # Execute valid mutants
        for i, m in enumerate(valid_mutants, 1):
            mutant_result = self._test_mutant(m, inputs, expected_outputs, i, len(valid_mutants))
            results.append(mutant_result)
            
            if mutant_result.killed_by:
                kill_stats[mutant_result.killed_by] = kill_stats.get(mutant_result.killed_by, 0) + 1
        
        # Build report
        elapsed = time.time() - start_time
        
        killed = len([r for r in results if r.status == "killed"])
        survived = len([r for r in results if r.status == "survived"])
        runtime_errors = len([r for r in results if r.status == "runtime_error"])
        timeouts = len([r for r in results if r.status == "timeout"])
        invalid = len([r for r in results if r.status == "invalid"])
        equivalent = len([r for r in results if r.status == "equivalent"])
        
        denominator = killed + survived + runtime_errors + timeouts
        mutation_score = killed / denominator if denominator > 0 else 0.0
        
        self.report = MutationReport(
            total_mutants=len(self.mutants),
            killed=killed,
            survived=survived,
            runtime_errors=runtime_errors,
            timeouts=timeouts,
            invalid=invalid,
            equivalent=equivalent,
            mutation_score=mutation_score,
            time_taken_seconds=elapsed,
            results=results,
            input_patterns=list(inputs.keys())
        )
        
        # Store kill stats for reporting
        self._kill_stats = kill_stats
        
        return self.report
    
    def _test_mutant(
        self,
        mutant: Dict,
        inputs: Dict[str, str],
        expected_outputs: Dict[str, str],
        current: int,
        total: int
    ) -> MutantResult:
        """Test a single mutant against all inputs."""
        mutant_id = mutant["id"]
        mutant_source = mutant["source"]
        locations_str = mutant.get("locations_str")  # For combined mutations
        
        # Check if mutant is equivalent (no semantic changes)
        if self._is_equivalent_mutant(mutant_source):
            self._print_progress(current, total, mutant_id, "equivalent", "none")
            return MutantResult(
                id=mutant_id,
                status="equivalent",
                pattern=mutant["pattern"],
                location=mutant["location"],
                locations_str=locations_str,
                error_message="Equivalent mutation (no semantic change)",
                mutant_source=mutant_source
            )
        
        for input_name, stdin_input in inputs.items():
            expected_output = expected_outputs[input_name]
            
            result = SafeExecutor.execute(mutant_source, stdin_input=stdin_input, timeout=self.timeout)
            
            if result["status"] == "timeout":
                self._print_progress(current, total, mutant_id, "timeout", input_name)
                return MutantResult(
                    id=mutant_id,
                    status="timeout",
                    pattern=mutant["pattern"],
                    location=mutant["location"],
                    locations_str=locations_str,
                    killed_by=input_name,
                    mutant_source=mutant_source
                )
            
            if result["status"] == "runtime_error":
                self._print_progress(current, total, mutant_id, "runtime_error", input_name)
                return MutantResult(
                    id=mutant_id,
                    status="runtime_error",
                    pattern=mutant["pattern"],
                    location=mutant["location"],
                    locations_str=locations_str,
                    stderr=result["stderr"],
                    killed_by=input_name,
                    mutant_source=mutant_source
                )
            
            if not _outputs_equal(result["stdout"], expected_output):
                self._print_progress(current, total, mutant_id, "killed", input_name)
                return MutantResult(
                    id=mutant_id,
                    status="killed",
                    pattern=mutant["pattern"],
                    location=mutant["location"],
                    locations_str=locations_str,
                    original_output=expected_output,
                    mutant_output=result["stdout"],
                    killed_by=input_name,
                    mutant_source=mutant_source
                )
        
        # Survived all inputs
        self._print_progress(current, total, mutant_id, "survived", "all")
        return MutantResult(
            id=mutant_id,
            status="survived",
            pattern=mutant["pattern"],
            location=mutant["location"],
            locations_str=locations_str,
            mutant_source=mutant_source
        )
    
    def _is_equivalent_mutant(self, mutant_source: str) -> bool:
        """
        Check if mutant is equivalent (no semantic changes).
        
        This is a heuristic that looks for:
        1. No actual code changes (just formatting)
        2. Changes that don't affect semantics (e.g., swapping commutative operands)
        """
        # Get changed lines between original and mutant
        changed_lines = DiffGenerator.get_changed_lines(self.original_source, mutant_source)
        
        if not changed_lines:
            return True  # No changes at all
        
        # Check if changes are only cosmetic
        import difflib
        original_lines = self.original_source.splitlines()
        mutant_lines = mutant_source.splitlines()
        
        differ = difflib.SequenceMatcher(None, original_lines, mutant_lines)
        has_meaningful_changes = False
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag in ('replace', 'delete', 'insert'):
                orig = original_lines[i1:i2]
                mut = mutant_lines[j1:j2]
                
                # Check if it's a meaningful change (not just cosmetic)
                if not self._is_cosmetic_change(orig, mut):
                    has_meaningful_changes = True
                    break
        
        return not has_meaningful_changes
    
    def _print_progress(self, current: int, total: int, mutant_id: str, status: str, input_name: str):
        """Print progress line with appropriate color."""
        status_colors = {
            "killed": Colors.GREEN,
            "survived": Colors.RED,
            "runtime_error": Colors.YELLOW,
            "timeout": Colors.MAGENTA,
            "equivalent": Colors.GRAY
        }
        color = status_colors.get(status, Colors.GRAY)
        
        by_str = f" (by {input_name})" if input_name != "all" and input_name != "none" else ""
        print(f"  [{current}/{total}] {mutant_id}: {color}{status}{Colors.RESET}{by_str}")
    
    # ========================================================================
    # REPORTING (keep existing methods but update for equivalent status)
    # ========================================================================
    
    def print_report(self, output_mode: str = "normal", filter_status: str = "all"):
        """
        Print mutation testing report.
        
        Args:
            output_mode: "minimal", "normal", or "verbose"
            filter_status: "all", "survived", "killed", or "equivalent"
        """
        if not self.report:
            print("No results. Run mutation testing first.")
            return
        
        r = self.report
        
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}MUTATION TESTING REPORT{Colors.RESET}")
        print(f"{'='*60}\n")
        
        # Summary
        print(f"{Colors.GREEN}Killed:         {r.killed}{Colors.RESET}")
        print(f"{Colors.RED}Survived:       {r.survived}{Colors.RESET}")
        print(f"{Colors.YELLOW}Runtime errors: {r.runtime_errors}{Colors.RESET}")
        print(f"{Colors.MAGENTA}Timeouts:       {r.timeouts}{Colors.RESET}")
        print(f"{Colors.GRAY}Invalid:        {r.invalid}{Colors.RESET}")
        print(f"{Colors.GRAY}Equivalent:     {r.equivalent}{Colors.RESET}")
        print()
        
        # Score
        denominator = r.killed + r.survived + r.runtime_errors + r.timeouts
        if denominator > 0:
            score = r.killed / denominator
            score_color = Colors.GREEN if score >= 0.8 else Colors.YELLOW if score >= 0.6 else Colors.RED
            print(f"{Colors.BOLD}Score: {score_color}{r.killed}/{denominator} = {score*100:.1f}%{Colors.RESET}")
        else:
            print(f"{Colors.BOLD}Score: N/A (no valid mutants){Colors.RESET}")
        
        print(f"Time:  {r.time_taken_seconds:.2f}s")
        
        # Kill breakdown (if multi-input)
        if hasattr(self, '_kill_stats') and self._kill_stats and len(self._kill_stats) > 1:
            print(f"\n{Colors.BOLD}Kill Breakdown by Input{Colors.RESET}")
            print(f"{'─'*40}")
            for input_name, count in sorted(self._kill_stats.items(), key=lambda x: -x[1]):
                bar = "█" * min(count, 20)
                print(f"  {input_name:<20} {count:>3} {Colors.GREEN}{bar}{Colors.RESET}")
        
        # Detailed results (based on output_mode and filter)
        if output_mode == "minimal":
            return
        
        # Filter results
        if filter_status == "survived":
            show_results = [res for res in r.results if res.status == "survived"]
        elif filter_status == "killed":
            show_results = [res for res in r.results if res.status == "killed"]
        elif filter_status == "equivalent":
            show_results = [res for res in r.results if res.status == "equivalent"]
        else:
            # For "all" with normal mode, only show survived (weaknesses)
            if output_mode == "normal":
                show_results = [res for res in r.results if res.status in ("survived", "equivalent")]
            else:
                show_results = [res for res in r.results if res.status in ("killed", "survived", "equivalent")]
        
        if show_results:
            print(f"\n{Colors.BOLD}{'─'*60}{Colors.RESET}")
            
            for res in show_results:
                show_diff = (output_mode == "verbose" or filter_status != "all" 
                             or res.status in ("survived", "equivalent"))
                self._print_mutant_detail(res, show_diff=show_diff)
    
    def _print_mutant_detail(self, res: MutantResult, show_diff: bool = True):
        """Print details for a single mutant result with code diff."""
        status_colors = {
            "killed": Colors.GREEN,
            "survived": Colors.RED,
            "runtime_error": Colors.YELLOW,
            "timeout": Colors.MAGENTA,
            "equivalent": Colors.GRAY,
            "invalid": Colors.GRAY
        }
        color = status_colors.get(res.status, Colors.GRAY)
        
        print(f"\n{Colors.BOLD}[{color}{res.status.upper()}{Colors.RESET}{Colors.BOLD}] {res.id}{Colors.RESET}")
        
        if res.pattern:
            print(f"  Pattern: {res.pattern}")
        
        # Show locations - prefer locations_str for combined mutations
        if res.locations_str:
            print(f"  Locations: {res.locations_str}")
        elif res.location:
            print(f"  Location: line {res.location[0]}, col {res.location[1]}")
        
        if res.killed_by:
            print(f"  Killed by: {Colors.CYAN}{res.killed_by}{Colors.RESET}")
        if res.error_message:
            print(f"  Error: {res.error_message[:100]}")
        
        if res.status == "survived":
            print(f"  {Colors.RED}⚠ Test suite did not detect this mutation!{Colors.RESET}")
        elif res.status == "equivalent":
            print(f"  {Colors.GRAY}⚠ Equivalent mutation (no semantic change){Colors.RESET}")
        
        # Show output comparison for killed mutants
        if res.status == "killed" and res.original_output is not None and res.mutant_output is not None:
            self._print_output_comparison(res.original_output, res.mutant_output)
        
        # Show stderr for runtime errors
        if res.status == "runtime_error" and res.stderr:
            print(f"\n  {Colors.DIM}─── Error Output ───{Colors.RESET}")
            stderr_lines = res.stderr.strip().split('\n')
            for line in stderr_lines[:5]:  # Limit to 5 lines
                print(f"  {Colors.YELLOW}{line[:80]}{Colors.RESET}")
            if len(stderr_lines) > 5:
                print(f"  {Colors.DIM}... ({len(stderr_lines) - 5} more lines){Colors.RESET}")
        
        # Show code diff
        if show_diff and res.mutant_source:
            self._print_code_diff(res)
    
    def _print_output_comparison(self, original_output: str, mutant_output: str):
        """Print side-by-side comparison of original vs mutant output."""
        print(f"\n  {Colors.DIM}─── Output Comparison ───{Colors.RESET}")
        
        orig_lines = original_output.strip().split('\n') if original_output else ["(empty)"]
        mut_lines = mutant_output.strip().split('\n') if mutant_output else ["(empty)"]
        
        # Limit display to reasonable length
        max_lines = 8
        orig_truncated = len(orig_lines) > max_lines
        mut_truncated = len(mut_lines) > max_lines
        
        orig_display = orig_lines[:max_lines]
        mut_display = mut_lines[:max_lines]
        
        # Find which lines differ
        max_len = max(len(orig_display), len(mut_display))
        
        print(f"  {Colors.GREEN}Original:{Colors.RESET}")
        for i, line in enumerate(orig_display):
            # Truncate long lines
            display_line = line[:70] + "..." if len(line) > 70 else line
            # Check if this line differs
            mut_line = mut_display[i] if i < len(mut_display) else None
            if mut_line is None or line != mut_line:
                print(f"    {Colors.GREEN}{display_line}{Colors.RESET}")
            else:
                print(f"    {Colors.DIM}{display_line}{Colors.RESET}")
        if orig_truncated:
            print(f"    {Colors.DIM}... ({len(orig_lines) - max_lines} more lines){Colors.RESET}")
        
        print(f"  {Colors.RED}Mutant:{Colors.RESET}")
        for i, line in enumerate(mut_display):
            # Truncate long lines
            display_line = line[:70] + "..." if len(line) > 70 else line
            # Check if this line differs
            orig_line = orig_display[i] if i < len(orig_display) else None
            if orig_line is None or line != orig_line:
                print(f"    {Colors.RED}{display_line}{Colors.RESET}")
            else:
                print(f"    {Colors.DIM}{display_line}{Colors.RESET}")
        if mut_truncated:
            print(f"    {Colors.DIM}... ({len(mut_lines) - max_lines} more lines){Colors.RESET}")
    
    # ========================================================================
    # EXISTING METHODS (keep from original)
    # ========================================================================
    
    def _print_code_diff(self, res: MutantResult, context: int = 1):
        """Print a clear diff showing original vs mutated code at each location."""
        # Keep existing implementation from your original code
        if not res.mutant_source:
            return
        
        original_lines = self.original_source.splitlines()
        mutant_lines = res.mutant_source.splitlines()
        
        # Find all actual differences between original and mutant
        import difflib
        differ = difflib.SequenceMatcher(None, original_lines, mutant_lines)
        
        # Filter out cosmetic changes (whitespace, quotes, comments)
        meaningful_changes = []
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag in ('replace', 'delete', 'insert'):
                orig = original_lines[i1:i2]
                mut = mutant_lines[j1:j2]
                
                # Skip if it's just cosmetic
                if self._is_cosmetic_change(orig, mut):
                    continue
                
                meaningful_changes.append({
                    'tag': tag,
                    'orig_start': i1, 'orig_end': i2,
                    'mut_start': j1, 'mut_end': j2,
                    'orig_lines': orig,
                    'mut_lines': mut
                })
        
        if not meaningful_changes:
            # If no meaningful changes, the mutation might be quote/whitespace only
            print(f"  {Colors.DIM}(only formatting changes){Colors.RESET}")
            return
        
        # Parse pattern names for combined mutations
        pattern_parts = res.pattern.split("+") if res.pattern and "+" in res.pattern else [res.pattern]
        
        print(f"  {Colors.DIM}─── Code Changes ───{Colors.RESET}")
        
        # Show each change
        for i, change in enumerate(meaningful_changes[:5]):  # Limit to 5 changes
            # Get description for this change
            pattern_name = pattern_parts[i] if i < len(pattern_parts) else None
            desc = self._get_mutation_description(pattern_name) if pattern_name else "modified"
            
            orig_ln = change['orig_start'] + 1
            
            print(f"\n  {Colors.BOLD}Line {orig_ln}{Colors.RESET} ({desc}):")
            
            # Show context before (from original)
            ctx_start = max(0, change['orig_start'] - context)
            for idx in range(ctx_start, change['orig_start']):
                print(f"  {Colors.DIM}  {idx+1:3}│ {original_lines[idx]}{Colors.RESET}")
            
            # Show the actual change
            if change['tag'] == 'replace':
                for idx, line in enumerate(change['orig_lines']):
                    ln = change['orig_start'] + idx + 1
                    print(f"  {Colors.RED}  {ln:3}│-{line}{Colors.RESET}")
                for line in change['mut_lines']:
                    print(f"  {Colors.GREEN}     │+{line}{Colors.RESET}")
            elif change['tag'] == 'delete':
                for idx, line in enumerate(change['orig_lines']):
                    ln = change['orig_start'] + idx + 1
                    print(f"  {Colors.RED}  {ln:3}│-{line}{Colors.RESET}")
            elif change['tag'] == 'insert':
                for line in change['mut_lines']:
                    print(f"  {Colors.GREEN}     │+{line}{Colors.RESET}")
            
            # Show context after (from original)
            ctx_end = min(len(original_lines), change['orig_end'] + context)
            for idx in range(change['orig_end'], ctx_end):
                print(f"  {Colors.DIM}  {idx+1:3}│ {original_lines[idx]}{Colors.RESET}")
        
        if len(meaningful_changes) > 5:
            print(f"\n  {Colors.DIM}  ... and {len(meaningful_changes) - 5} more changes{Colors.RESET}")
    
    def _is_cosmetic_change(self, orig_lines: List[str], mut_lines: List[str]) -> bool:
        """Check if a change is purely cosmetic (whitespace, quotes, comments)."""
        # Keep existing implementation from your original code
        def normalize(lines):
            normalized = []
            for line in lines:
                # Skip empty/whitespace-only lines
                if not line.strip():
                    continue
                # Skip comment-only lines
                if line.strip().startswith('#'):
                    continue
                # Normalize quotes: replace double with single
                norm = line.replace('"', "'")
                # Remove trailing whitespace
                norm = norm.rstrip()
                normalized.append(norm)
            return normalized
        
        norm_orig = normalize(orig_lines)
        norm_mut = normalize(mut_lines)
        
        return norm_orig == norm_mut
    
    def _get_mutation_description(self, pattern: str) -> str:
        """Get a short description of what the mutation does."""
        # Keep existing implementation from your original code
        if not pattern:
            return "modified"
        descriptions = {
            "arithmetic_op_replacement": "+ → - or * → /",
            "relational_op_replacement": "!= ↔ ==, < ↔ <=",
            "relational_op_all": "< → <= → > → >=",
            "logical_op_replacement": "and ↔ or",
            "swap_binary_operands": "a op b → b op a",
            "if_else_negation": "swapped if/else",
            "off_by_one": "range(n) → range(n±1)",
            "zero_one_swap": "0 ↔ 1",
            "bool_constant_swap": "True ↔ False",
            "return_to_none": "return x → None",
            "remove_not": "not x → x",
            "statement_deletion": "deleted line",
            "remove_if_body": "if body → pass",
            "remove_else_body": "else body → pass",
            "while_to_false": "while → False",
            "for_to_empty": "for → empty []",
            "remove_break": "break → pass",
            "remove_continue": "continue → pass",
            "negate_unary": "+x ↔ -x",
            "is_op_replacement": "is ↔ is not",
            "in_op_replacement": "in ↔ not in",
            "empty_list_mutation": "[] → [1]",
            "empty_dict_mutation": "{} → {'k': 1}",
            "return_none_to_one": "None → 1",
            "remove_raise": "raise → pass",
        }
        return descriptions.get(pattern, pattern)
    
    def save_report(self, filepath: str):
        """Save report to JSON file."""
        if not self.report:
            print("No results to save.")
            return
        
        with open(filepath, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def get_survived(self) -> List[MutantResult]:
        """Get list of survived mutants."""
        if not self.report:
            return []
        return [r for r in self.report.results if r.status == "survived"]
    
    def get_killed(self) -> List[MutantResult]:
        """Get list of killed mutants."""
        if not self.report:
            return []
        return [r for r in self.report.results if r.status == "killed"]
    
    def get_equivalent(self) -> List[MutantResult]:
        """Get list of equivalent mutants."""
        if not self.report:
            return []
        return [r for r in self.report.results if r.status == "equivalent"]


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_mutation_testing(
    source_code: str,
    mutation_results: List[Any],
    inputs: Union[str, Dict[str, str]],
    timeout: int = 3,
    save_to: Optional[str] = None
) -> MutationReport:
    """
    Convenience function to run mutation testing.
    
    Args:
        source_code: Original program source code
        mutation_results: List of MutationResult from engine.mutate()
        inputs: Either a single stdin string, or dict of {name: stdin_string}
        timeout: Per-mutant timeout
        save_to: Optional filepath to save JSON results
        
    Returns:
        MutationReport
    """
    runner = MutationRunner(source_code, timeout=timeout)
    runner.add_mutants(mutation_results)
    
    if isinstance(inputs, str):
        report = runner.run(inputs)
    else:
        report = runner.run_multi(inputs)
    
    runner.print_report()
    
    if save_to:
        runner.save_report(save_to)
    
    return report