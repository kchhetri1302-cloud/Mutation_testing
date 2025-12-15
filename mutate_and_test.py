#!/usr/bin/env python3
"""
Mutation Testing CLI
====================

A streamlined mutation testing tool.

Usage:
    mutate prog.py -i input.txt              # Test with input file
    mutate prog.py --auto                    # Auto-discover inputs
    mutate prog.py --auto --behavior-only    # Only behavior-changing mutations
    mutate prog.py --auto -k 3               # Up to 3rd-order mutations
    mutate --init                            # Initialize test_inputs/
    mutate --list                            # List mutations and patterns

Simplified from the original, this CLI focuses on:
- Two input modes: manual (-i) or auto-discovery (--auto)
- Three output modes: minimal, normal, verbose
- Clear mutation selection: behavior-only or specific indices
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add src to path
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Import modules
from common import Colors, parse_scope
from mutation_engine import (
    MutationEngine, 
    MutationCombiner,
    BEHAVIOR_CHANGING,
    SEMANTIC_PRESERVING
)
from mutation_runner import MutationRunner
from input_discovery import InputDiscovery, initialize_test_inputs


# ============================================================================
# DIRECTORIES
# ============================================================================

PROGRAMS_DIR = SCRIPT_DIR / "programs"
RESULTS_DIR = SCRIPT_DIR / "results"
TEST_INPUTS_DIR = SCRIPT_DIR / "test_inputs"
MUTANTS_DIR = SCRIPT_DIR / "mutants"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Get timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# LIST COMMAND
# ============================================================================

def list_info():
    """List all mutations and input patterns."""
    engine = MutationEngine()
    
    print(f"\n{Colors.BOLD}AVAILABLE MUTATIONS{Colors.RESET}")
    print(f"{'─'*60}\n")
    
    # Behavior-changing
    print(f"{Colors.GREEN}Behavior-Changing (recommended for testing):{Colors.RESET}")
    for idx in BEHAVIOR_CHANGING:
        name = engine.get_mutation_name(idx)
        if name:
            print(f"  {idx:2}: {name}")
    
    print()
    
    # Semantic-preserving
    print(f"{Colors.YELLOW}Semantic-Preserving (refactoring):{Colors.RESET}")
    for idx in SEMANTIC_PRESERVING:
        name = engine.get_mutation_name(idx)
        if name:
            print(f"  {idx:2}: {name}")
    
    # Input patterns
    print(f"\n{Colors.BOLD}INPUT PATTERNS{Colors.RESET}")
    print(f"{'─'*60}")
    
    patterns_file = TEST_INPUTS_DIR / "patterns.json"
    if patterns_file.exists():
        with open(patterns_file) as f:
            patterns = json.load(f)
        print(f"Directory: {TEST_INPUTS_DIR}\n")
        for name, info in patterns.items():
            print(f"  {Colors.CYAN}{name}{Colors.RESET}: {info.get('description', 'N/A')}")
    else:
        print(f"  {Colors.YELLOW}Not initialized. Run: mutate --init{Colors.RESET}")
    
    print()


# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def run_test(
    program_path: Path,
    input_file: Optional[str],
    auto_discover: bool,
    behavior_only: bool,
    mutations: Optional[List[int]],
    scope: Optional[str],
    k_order: int,
    timeout: int,
    workers: int,
    output_mode: str,
    filter_status: str,
    save_mutants: bool,
) -> Optional[int]:
    """
    Run mutation testing on a program.
    
    Returns:
        0 on success, 1 on error, 2 on weak test suite (score < 60%)
    """
    timestamp = get_timestamp()
    
    # Load source code
    if not program_path.exists():
        # Try programs directory
        program_path = PROGRAMS_DIR / program_path.name
        if not program_path.exists():
            print(f"{Colors.RED}Error: Program not found: {program_path}{Colors.RESET}")
            return 1
    
    source_code = program_path.read_text()
    program_name = program_path.stem
    total_lines = len(source_code.splitlines())
    
    print(f"\n{Colors.BOLD}{'═'*60}{Colors.RESET}")
    print(f"{Colors.BOLD}MUTATION TESTING: {program_name}.py{Colors.RESET}")
    print(f"{'═'*60}")
    print(f"Source: {program_path} ({total_lines} lines)")
    
    # ========================================================================
    # INPUT HANDLING
    # ========================================================================
    
    inputs: Dict[str, str] = {}
    
    if auto_discover:
        # Auto-discover inputs from patterns.json
        if not TEST_INPUTS_DIR.exists():
            print(f"{Colors.RED}Error: test_inputs/ not found{Colors.RESET}")
            print(f"Run: mutate --init")
            return 1
        
        discovery = InputDiscovery(str(TEST_INPUTS_DIR), timeout=timeout)
        result = discovery.discover(source_code, verbose=(output_mode != "minimal"))
        
        if not result.success:
            print(f"{Colors.RED}Error: {result.error}{Colors.RESET}")
            print(f"Try using -i <input_file> instead")
            return 1
        
        inputs = discovery.load_inputs(result.patterns)
        print(f"\nUsing {len(inputs)} input pattern(s)")
        
    elif input_file:
        # Manual input file
        input_path = Path(input_file)
        if not input_path.exists():
            input_path = program_path.parent / input_file
        
        if not input_path.exists():
            print(f"{Colors.RED}Error: Input file not found: {input_file}{Colors.RESET}")
            return 1
        
        inputs = {"manual": input_path.read_text()}
        print(f"Input: {input_path}")
        
    else:
        print(f"{Colors.YELLOW}Warning: No input specified. Use -i or --auto{Colors.RESET}")
        inputs = {"empty": ""}
    
    # ========================================================================
    # MUTATION GENERATION
    # ========================================================================
    
    engine = MutationEngine()
    
    # Determine line range from scope
    line_range = None
    if scope:
        line_range = parse_scope(scope, total_lines)
        print(f"Scope: lines {line_range[0]}-{line_range[1]}")
    
    # Determine mutations to use
    if mutations:
        mutations_to_apply = mutations
        print(f"Mutations: {mutations}")
    elif behavior_only:
        mutations_to_apply = BEHAVIOR_CHANGING
        print(f"Category: behavior-changing ({len(BEHAVIOR_CHANGING)} patterns)")
    else:
        mutations_to_apply = None
        print(f"Category: all applicable")
    
    # Generate mutations
    is_higher_order = k_order > 1
    
    if is_higher_order:
        print(f"Order: up to k={k_order}")
        
        combiner = MutationCombiner(engine)
        applicable = engine.get_applicable(source_code)
        
        if mutations_to_apply:
            use_mutations = [m for m in mutations_to_apply if m in applicable]
        else:
            use_mutations = applicable

        mutation_results = []
        for k in range(1, k_order + 1):
            k_results = combiner.generate_combinations_from_mutants(
                code=source_code,
                k=k,
                mutations_to_use=use_mutations,
                max_combinations=1000,  # Increase limit
                line_range=line_range
            )
            mutation_results.extend(k_results)
        
        print(f"\nGenerated {len(mutation_results)} mutations (k=1 to k={k_order})")
            
    else:
        mutation_results = engine.mutate(
            source_code,
            mutations=mutations_to_apply,
            line_range=line_range,
            behavior_only=behavior_only
        )
        
        valid_count = len([m for m in mutation_results if m.is_valid])
        print(f"\nGenerated {len(mutation_results)} mutations ({valid_count} valid)")
    
    # Save mutants if requested
    if save_mutants and mutation_results:
        mutant_folder = ensure_dir(MUTANTS_DIR / f"{program_name}_{timestamp}")
        saved = 0
        for i, m in enumerate(mutation_results):
            if m.is_valid:
                if hasattr(m, 'mutation_names'):
                    pattern = "_".join(m.mutation_names[:2])
                else:
                    pattern = m.pattern_name
                pattern_safe = pattern.replace("/", "_").replace("\\", "_")[:30]
                mid = getattr(m, 'mutation_id', None) or f"mutant_{i+1:04d}"
                filepath = mutant_folder / f"{mid}_{pattern_safe}.py"
                filepath.write_text(m.mutated_code)
                saved += 1
        print(f"Saved {saved} mutants to: {mutant_folder}")
    
    # ========================================================================
    # MUTATION TESTING
    # ========================================================================
    
    runner = MutationRunner(source_code, timeout=timeout, max_workers=workers)
    runner.add_mutants(mutation_results)
    
    try:
        if len(inputs) > 1:
            report = runner.run_multi(inputs)
        else:
            report = runner.run(list(inputs.values())[0])
    except RuntimeError as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        return 1
    
    # Print report
    runner.print_report(output_mode=output_mode, filter_status=filter_status)
    
    # Save results
    ensure_dir(RESULTS_DIR)
    result_file = RESULTS_DIR / f"{program_name}_{timestamp}.json"
    runner.save_report(str(result_file))
    
    # Return code based on score
    if report.mutation_score < 0.6:
        return 2  # Weak test suite
    return 0


# ============================================================================
# BATCH MODE
# ============================================================================

def run_batch(
    pattern: str,
    auto_discover: bool,
    behavior_only: bool,
    timeout: int,
    workers: int,
):
    """Run mutation testing on all programs matching pattern."""
    programs = sorted(PROGRAMS_DIR.glob(pattern))
    
    if not programs:
        print(f"{Colors.YELLOW}No programs found in {PROGRAMS_DIR} matching '{pattern}'{Colors.RESET}")
        return
    
    print(f"\n{Colors.BOLD}BATCH MUTATION TESTING{Colors.RESET}")
    print(f"Programs: {len(programs)}")
    print()
    
    successful = []    # Score >= 60%
    weak_tests = []    # Score < 60% (return code 2)
    failed = []        # Errors (return code 1)
    
    for i, program in enumerate(programs, 1):
        print(f"\n{Colors.CYAN}[{i}/{len(programs)}] {program.name}{Colors.RESET}")
        
        try:
            ret = run_test(
                program_path=program,
                input_file=None,
                auto_discover=auto_discover,
                behavior_only=behavior_only,
                mutations=None,
                scope=None,
                k_order=1,
                timeout=timeout,
                workers=workers,
                output_mode="minimal",
                filter_status="all",
                save_mutants=False,
            )
            
            if ret == 0:
                successful.append(program.name)
            elif ret == 2:
                weak_tests.append(program.name)
            else:
                failed.append({"program": program.name, "reason": "Unknown error"})
            
        except Exception as e:
            error_msg = str(e)[:60]
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            failed.append({"program": program.name, "reason": error_msg})
    
    # Summary
    total = len(programs)
    print(f"\n{Colors.BOLD}{'═'*60}{Colors.RESET}")
    print(f"{Colors.BOLD}BATCH SUMMARY{Colors.RESET}")
    print(f"{'═'*60}")
    
    print(f"\n{Colors.GREEN}✓ Passed (score ≥ 60%): {len(successful)}/{total}{Colors.RESET}")
    print(f"{Colors.YELLOW}⚠ Weak tests (score < 60%): {len(weak_tests)}/{total}{Colors.RESET}")
    print(f"{Colors.RED}✗ Failed/Errors: {len(failed)}/{total}{Colors.RESET}")
    
    # Show weak test programs
    if weak_tests:
        print(f"\n{Colors.BOLD}Programs with Weak Test Coverage:{Colors.RESET}")
        print(f"{Colors.DIM}(Mutation score < 60% - many mutants survived){Colors.RESET}")
        print(f"{'─'*60}")
        for name in weak_tests:
            print(f"  {Colors.YELLOW}• {name}{Colors.RESET}")
    
    # Show failed programs
    if failed:
        print(f"\n{Colors.BOLD}Failed Programs:{Colors.RESET}")
        print(f"{'─'*60}")
        for item in failed:
            print(f"  {Colors.RED}• {item['program']}{Colors.RESET}")
            print(f"    Reason: {item['reason']}")
    
    print()


# ============================================================================
# MAIN
# ============================================================================

class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter for better help display with proper alignment."""
    def __init__(self, prog):
        super().__init__(prog, max_help_position=30, width=90)


def main():
    parser = argparse.ArgumentParser(
        prog="mutate",
        description="Mutation Testing Tool",
        formatter_class=CustomHelpFormatter,
        epilog="""
Examples:
  mutate prog.py -i input.txt              # Test with manual input
  mutate prog.py --auto                    # Auto-discover inputs
  mutate prog.py --auto --behavior-only    # Behavior-changing mutations only
  mutate prog.py --auto -k 3               # Up to 3rd-order mutations
  mutate prog.py -m 24 25 28               # Specific mutation indices
  mutate prog.py --scope "50%"             # First 50% of code
  mutate prog.py --scope "10-20"           # Lines 10-20 only
  mutate --init                            # Initialize test_inputs/
  mutate --list                            # List mutations and patterns
  mutate --batch --auto                    # Test all programs in programs/
"""
    )
    
    # Positional
    parser.add_argument("program", nargs="?", help="Program file to test")
    
    # Input
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("-i", "--input", metavar="INPUT",
                             help="Input file for stdin")
    input_group.add_argument("--auto", action="store_true", 
                             help="Auto-discover inputs from test_inputs/")
    
    # Mutations
    mutation_group = parser.add_argument_group("Mutations")
    mutation_group.add_argument("--behavior-only", action="store_true",
                                help="Only behavior-changing mutations (recommended)")
    mutation_group.add_argument("-m", "--mutations", type=int, nargs="+", 
                                metavar="MUTATIONS",
                                help="Specific mutation indices")
    mutation_group.add_argument("--scope", metavar="SCOPE",
                                help="Code scope: '50%%' (percentage) or '10-20' (line range)")
    mutation_group.add_argument("-k", type=int, default=1,
                                help="Generate up to k-order mutations (default: 1)")
    
    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", choices=["minimal", "normal", "verbose"],
                              default="normal",
                              help="Output detail level")
    output_group.add_argument("--filter", choices=["all", "survived", "killed"],
                              default="all",
                              help="Filter results by status")
    output_group.add_argument("--no-color", action="store_true", 
                              help="Disable colors")
    output_group.add_argument("--save-mutants", action="store_true",
                              help="Save mutant files to mutants/")
    
    # Execution
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument("-t", "--timeout", type=int, default=3,
                            help="Timeout per mutant (seconds)")
    exec_group.add_argument("-w", "--workers", type=int, default=4,
                            help="Parallel workers")
    
    # Commands
    cmd_group = parser.add_argument_group("Commands")
    cmd_group.add_argument("--init", action="store_true",
                           help="Initialize test_inputs/ with defaults")
    cmd_group.add_argument("--list", action="store_true",
                           help="List mutations and input patterns")
    cmd_group.add_argument("--batch", action="store_true",
                           help="Run on all programs in programs/")
    cmd_group.add_argument("--pattern", default="*.py",
                           help="File pattern for batch mode")
    
    args = parser.parse_args()
    
    # Handle colors
    if args.no_color:
        Colors.disable()
    
    # Command: --init
    if args.init:
        print(f"\n{Colors.BOLD}Initializing test inputs...{Colors.RESET}\n")
        ensure_dir(TEST_INPUTS_DIR)
        initialize_test_inputs(str(TEST_INPUTS_DIR))
        return 0
    
    # Command: --list
    if args.list:
        list_info()
        return 0
    
    # Command: --batch
    if args.batch:
        ensure_dir(PROGRAMS_DIR)
        run_batch(
            pattern=args.pattern,
            auto_discover=args.auto,
            behavior_only=args.behavior_only,
            timeout=args.timeout,
            workers=args.workers,
        )
        return 0
    
    # Main: test a program
    if not args.program:
        parser.print_help()
        print(f"\n{Colors.YELLOW}Tip: Use --init to set up, --list to see options{Colors.RESET}")
        return 0
    
    program_path = Path(args.program)
    
    result = run_test(
        program_path=program_path,
        input_file=args.input,
        auto_discover=args.auto,
        behavior_only=args.behavior_only,
        mutations=args.mutations,
        scope=args.scope,
        k_order=args.k,
        timeout=args.timeout,
        workers=args.workers,
        output_mode=args.output,
        filter_status=args.filter,
        save_mutants=args.save_mutants,
    )
    
    return result if result is not None else 1


if __name__ == "__main__":
    sys.exit(main() or 0)