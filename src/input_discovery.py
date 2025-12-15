"""
Input Pattern Discovery
=======================

Discovers what input pattern a program expects using configurable patterns.

All configuration comes from external files:
- patterns.json: Pattern definitions with probes
- *.txt: Test input files for each pattern

Usage:
    discovery = InputDiscovery("test_inputs")
    result = discovery.discover(source_code)
    
    if result.success:
        inputs = discovery.load_inputs(result.patterns)
        # Run mutation testing with inputs
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Import shared utilities
from common import SafeExecutor, Colors


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PatternMatch:
    """Result of matching a single pattern."""
    name: str
    file: str
    description: str
    probe: List[str]
    output: str
    runtime_ms: int


@dataclass
class DiscoveryResult:
    """Result of input pattern discovery."""
    success: bool
    patterns: List[PatternMatch] = field(default_factory=list)
    patterns_tried: int = 0
    error: Optional[str] = None
    
    @property
    def primary(self) -> Optional[PatternMatch]:
        """Get the first/primary matching pattern."""
        return self.patterns[0] if self.patterns else None


# ============================================================================
# INPUT DISCOVERY
# ============================================================================

class InputDiscovery:
    """
    Discovers valid input patterns for stdin-based programs.
    
    Probes the program with test inputs defined in patterns.json
    to determine which input formats the program accepts.
    
    Usage:
        discovery = InputDiscovery("test_inputs")
        
        # Find all matching patterns
        result = discovery.discover(source_code)
        
        if result.success:
            # Load test inputs for all matched patterns
            inputs = discovery.load_inputs(result.patterns)
            # inputs = {"pattern_name": "input_content", ...}
    """
    
    def __init__(self, test_inputs_dir: str = "test_inputs", timeout: int = 2):
        """
        Initialize discovery.
        
        Args:
            test_inputs_dir: Directory containing patterns.json and test files
            timeout: Timeout for each probe execution
        """
        self.test_inputs_dir = Path(test_inputs_dir)
        self.timeout = timeout
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict]:
        """Load patterns from patterns.json."""
        patterns_file = self.test_inputs_dir / "patterns.json"
        
        if not patterns_file.exists():
            raise FileNotFoundError(
                f"patterns.json not found at {patterns_file}\n"
                f"Run 'mutate --init' to create default files."
            )
        
        with open(patterns_file) as f:
            return json.load(f)
    
    def discover(self, code: str, verbose: bool = True) -> DiscoveryResult:
        """
        Discover all input patterns the program accepts.
        
        Args:
            code: Program source code
            verbose: Print progress
            
        Returns:
            DiscoveryResult with all matching patterns
        """
        if verbose:
            print(f"\n{Colors.BOLD}INPUT DISCOVERY{Colors.RESET}")
            print(f"Testing {len(self.patterns)} patterns...\n")
        
        patterns_tried = 0
        matched: List[PatternMatch] = []
        
        for pattern_name, pattern_info in self.patterns.items():
            patterns_tried += 1
            probe = pattern_info.get("probe", [])
            
            if not probe:
                if verbose:
                    print(f"  {Colors.GRAY}⊘ {pattern_name:<25} → No probe defined{Colors.RESET}")
                continue
            
            stdin_data = "\n".join(str(p) for p in probe)
            result = SafeExecutor.execute(code, stdin_input=stdin_data, timeout=self.timeout)
            
            if result["status"] == "success":
                if verbose:
                    preview = result["stdout"][:30].replace("\n", "\\n")
                    print(f"  {Colors.GREEN}✓ {pattern_name:<25} → {preview}...{Colors.RESET}")
                
                matched.append(PatternMatch(
                    name=pattern_name,
                    file=pattern_info.get("file", f"{pattern_name}.txt"),
                    description=pattern_info.get("description", ""),
                    probe=probe,
                    output=result["stdout"],
                    runtime_ms=result["runtime_ms"]
                ))
            else:
                if verbose:
                    status = "timeout" if result["status"] == "timeout" else "error"
                    print(f"  {Colors.RED}✗ {pattern_name:<25} → {status}{Colors.RESET}")
        
        if matched:
            if verbose:
                print(f"\n{Colors.BOLD}Matched: {len(matched)} pattern(s){Colors.RESET}")
            
            return DiscoveryResult(
                success=True,
                patterns=matched,
                patterns_tried=patterns_tried
            )
        else:
            if verbose:
                print(f"\n{Colors.RED}No matching patterns found{Colors.RESET}")
            
            return DiscoveryResult(
                success=False,
                patterns=[],
                patterns_tried=patterns_tried,
                error="No matching input pattern found"
            )
    
    def load_inputs(self, patterns: List[PatternMatch]) -> Dict[str, str]:
        """
        Load test input files for matched patterns.
        
        Splits each file into individual test cases based on probe length.
        For example, if probe is ["10", "5"] (2 lines), a file with 8 lines
        becomes 4 separate test cases.
        
        Args:
            patterns: List of PatternMatch objects
            
        Returns:
            Dict mapping input_name -> stdin_string
            Names are formatted as "pattern_name" for single case,
            or "pattern_name_1", "pattern_name_2", etc. for multiple cases.
        """
        inputs = {}
        
        for pattern in patterns:
            filepath = self.test_inputs_dir / pattern.file
            
            if not filepath.exists():
                # Fall back to probe values if file doesn't exist
                print(f"  {Colors.YELLOW}Warning: {pattern.file} not found, using probe{Colors.RESET}")
                inputs[pattern.name] = "\n".join(str(p) for p in pattern.probe)
                continue
            
            # Read file and split into lines
            content = filepath.read_text().strip()
            if not content:
                # Empty file - use probe values
                inputs[pattern.name] = "\n".join(str(p) for p in pattern.probe)
                continue
            
            lines = content.split('\n')
            lines_per_case = len(pattern.probe)
            
            if lines_per_case == 0:
                # No probe defined - treat entire file as one input
                inputs[pattern.name] = content
                continue
            
            # Split into test cases
            test_cases = []
            for i in range(0, len(lines), lines_per_case):
                chunk = lines[i:i + lines_per_case]
                if len(chunk) == lines_per_case:
                    test_cases.append('\n'.join(chunk))
            
            # Add to inputs dict
            if len(test_cases) == 0:
                # File too short - use probe values
                inputs[pattern.name] = "\n".join(str(p) for p in pattern.probe)
            elif len(test_cases) == 1:
                # Single test case - use pattern name directly
                inputs[pattern.name] = test_cases[0]
            else:
                # Multiple test cases - enumerate them
                for idx, case in enumerate(test_cases, 1):
                    inputs[f"{pattern.name}_{idx}"] = case
        
        return inputs
    
    def load_single_input(self, pattern_name: str) -> Optional[str]:
        """
        Load test input file for a single pattern.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Contents of the test file, or None if not found
        """
        if pattern_name not in self.patterns:
            return None
        
        filename = self.patterns[pattern_name].get("file", f"{pattern_name}.txt")
        filepath = self.test_inputs_dir / filename
        
        if filepath.exists():
            return filepath.read_text()
        return None
    
    def list_patterns(self) -> List[str]:
        """List all available pattern names."""
        return list(self.patterns.keys())
    
    def get_pattern_info(self, name: str) -> Optional[Dict]:
        """Get info about a specific pattern."""
        return self.patterns.get(name)


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_test_inputs(test_inputs_dir: str = "test_inputs"):
    """
    Create test_inputs folder with default patterns.json and test files.
    
    Args:
        test_inputs_dir: Directory to create
    """
    test_dir = Path(test_inputs_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Default patterns
    default_patterns = {
        "single_int": {
            "probe": ["5"],
            "file": "single_int.txt",
            "description": "Single integer input"
        },
        "single_float": {
            "probe": ["3.14"],
            "file": "single_float.txt",
            "description": "Single float input"
        },
        "single_string": {
            "probe": ["hello"],
            "file": "single_string.txt",
            "description": "Single string input"
        },
        "two_ints": {
            "probe": ["10", "5"],
            "file": "two_ints.txt",
            "description": "Two integer inputs"
        },
        "two_floats": {
            "probe": ["3.14", "2.71"],
            "file": "two_floats.txt",
            "description": "Two float inputs"
        },
        "three_ints": {
            "probe": ["10", "5", "3"],
            "file": "three_ints.txt",
            "description": "Three integer inputs"
        },
        "count_then_ints": {
            "probe": ["3", "10", "20", "30"],
            "file": "count_then_ints.txt",
            "description": "First input is count, then N integer values"
        },
        "count_then_floats": {
            "probe": ["3", "1.5", "2.5", "3.5"],
            "file": "count_then_floats.txt",
            "description": "First input is count, then N float values"
        },
        "count_ints_extra1": {
            "probe": ["3", "10", "20", "30", "2"],
            "file": "count_ints_extra1.txt",
            "description": "Count + N ints + 1 extra (e.g., index query)"
        },
        "multi_strings": {
            "probe": ["hello", "world", "test"],
            "file": "multi_strings.txt",
            "description": "Multiple string inputs"
        },
        "matrix_input": {
            "probe": ["2", "2", "1", "2", "3", "4"],
            "file": "matrix_input.txt",
            "description": "Matrix input (rows, cols, then values)"
        },
        # Edge case patterns - useful for killing more mutants
        "zero_and_positive": {
            "probe": ["0", "5"],
            "file": "zero_and_positive.txt",
            "description": "Zero and positive integer"
        },
        "negative_and_positive": {
            "probe": ["-4", "2"],
            "file": "negative_and_positive.txt",
            "description": "Negative and positive integer"
        },
        "two_zeros": {
            "probe": ["0", "0"],
            "file": "two_zeros.txt",
            "description": "Both inputs are zero"
        },
        "divide_by_zero": {
            "probe": ["10", "0"],
            "file": "divide_by_zero.txt",
            "description": "Division by zero test"
        },
        "float_division": {
            "probe": ["5", "2"],
            "file": "float_division.txt",
            "description": "Division that produces float result"
        },
        "even_number": {
            "probe": ["4"],
            "file": "even_number.txt",
            "description": "Even integer for is_even test"
        },
        "odd_number": {
            "probe": ["7"],
            "file": "odd_number.txt",
            "description": "Odd integer for is_even test"
        },
        "negative_even": {
            "probe": ["-6"],
            "file": "negative_even.txt",
            "description": "Negative even integer"
        },
        "factorial_zero": {
            "probe": ["0"],
            "file": "factorial_zero.txt",
            "description": "Factorial of 0 (edge case = 1)"
        },
        "factorial_one": {
            "probe": ["1"],
            "file": "factorial_one.txt",
            "description": "Factorial of 1"
        },
        "factorial_positive": {
            "probe": ["5"],
            "file": "factorial_positive.txt",
            "description": "Factorial of positive integer"
        },
        "factorial_negative": {
            "probe": ["-3"],
            "file": "factorial_negative.txt",
            "description": "Factorial of negative (should error)"
        },
    }
    
    # Default test files
    # Each file contains multiple test cases - the number of lines per case
    # matches the probe length defined in default_patterns above.
    # 
    # Format: lines are grouped by probe length
    # Example: two_ints has probe ["10", "5"] (2 lines), so each test case is 2 lines
    
    default_files = {
        # single_int: 1 line per case = 8 test cases
        "single_int.txt": "5\n42\n0\n-10\n100\n1\n-1\n999",
        
        # single_float: 1 line per case = 6 test cases  
        "single_float.txt": "3.14\n0.0\n-2.5\n100.0\n0.001\n-99.99",
        
        # single_string: 1 line per case = 5 test cases
        "single_string.txt": "hello\nworld\ntest\nfoo\nbar",
        
        # two_ints: 2 lines per case = 6 test cases
        # Covers: normal, zeros, negative, large, boundary (0,1), mixed signs
        "two_ints.txt": "\n".join([
            "10", "5",      # case 1: normal positive
            "0", "0",       # case 2: both zero
            "-5", "3",      # case 3: negative and positive
            "100", "1",     # case 4: large and small
            "0", "1",       # case 5: zero and one (boundary)
            "7", "7",       # case 6: equal values
        ]),
        
        # two_floats: 2 lines per case = 4 test cases
        "two_floats.txt": "\n".join([
            "3.14", "2.71",   # case 1: normal
            "0.0", "0.0",     # case 2: zeros
            "-1.5", "2.5",    # case 3: negative
            "1.0", "0.001",   # case 4: small divisor
        ]),
        
        # three_ints: 3 lines per case = 3 test cases
        "three_ints.txt": "\n".join([
            "10", "5", "3",     # case 1: normal
            "0", "0", "0",      # case 2: zeros
            "-1", "-2", "-3",   # case 3: negatives
        ]),
        
        # count_then_ints: 4 lines per case (count=3, then 3 values) = 2 test cases
        "count_then_ints.txt": "\n".join([
            "3", "10", "20", "30",    # case 1: normal
            "3", "0", "-5", "100",    # case 2: mixed values
        ]),
        
        # count_then_floats: 4 lines per case = 2 test cases
        "count_then_floats.txt": "\n".join([
            "3", "1.5", "2.5", "3.5",   # case 1: normal
            "3", "0.0", "-1.0", "0.5",  # case 2: with zero/negative
        ]),
        
        # count_ints_extra1: 5 lines per case = 2 test cases
        "count_ints_extra1.txt": "\n".join([
            "3", "10", "20", "30", "2",   # case 1: query index 2
            "3", "5", "15", "25", "0",    # case 2: query index 0
        ]),
        
        # multi_strings: 3 lines per case = 2 test cases
        "multi_strings.txt": "\n".join([
            "hello", "world", "test",  # case 1
            "foo", "bar", "baz",       # case 2
        ]),
        
        # matrix_input: 6 lines per case (2x2 matrix: rows, cols, 4 values) = 2 test cases
        "matrix_input.txt": "\n".join([
            "2", "2", "1", "2", "3", "4",       # case 1: 2x2 matrix
            "2", "2", "0", "0", "0", "0",       # case 2: zero matrix
        ]),
        
        # Edge case test files
        # zero_and_positive: 2 lines per case
        "zero_and_positive.txt": "\n".join([
            "0", "5",       # case 1
            "0", "1",       # case 2: zero and one
            "0", "100",     # case 3: zero and large
        ]),
        
        # negative_and_positive: 2 lines per case
        "negative_and_positive.txt": "\n".join([
            "-4", "2",      # case 1
            "-1", "1",      # case 2: -1 and 1
            "-10", "5",     # case 3
        ]),
        
        # two_zeros: 2 lines per case
        "two_zeros.txt": "\n".join([
            "0", "0",       # only case needed
        ]),
        
        # divide_by_zero: 2 lines per case
        "divide_by_zero.txt": "\n".join([
            "10", "0",      # case 1: positive / zero
            "-5", "0",      # case 2: negative / zero
            "0", "0",       # case 3: zero / zero
        ]),
        
        # float_division: 2 lines per case (divisions that don't yield integers)
        "float_division.txt": "\n".join([
            "5", "2",       # case 1: 2.5
            "7", "3",       # case 2: 2.333...
            "1", "3",       # case 3: 0.333...
        ]),
        
        # Single-int edge cases for is_even/factorial
        # even_number: 1 line per case
        "even_number.txt": "\n".join([
            "4", "0", "2", "-4", "100",
        ]),
        
        # odd_number: 1 line per case
        "odd_number.txt": "\n".join([
            "7", "1", "3", "-7", "99",
        ]),
        
        # negative_even: 1 line per case
        "negative_even.txt": "\n".join([
            "-6", "-2", "-100",
        ]),
        
        # factorial_zero: 1 line per case
        "factorial_zero.txt": "0",
        
        # factorial_one: 1 line per case
        "factorial_one.txt": "1",
        
        # factorial_positive: 1 line per case
        "factorial_positive.txt": "\n".join([
            "5", "3", "6", "10",
        ]),
        
        # factorial_negative: 1 line per case
        "factorial_negative.txt": "\n".join([
            "-3", "-1", "-10",
        ]),
    }
    
    # Write patterns.json
    patterns_file = test_dir / "patterns.json"
    with open(patterns_file, 'w') as f:
        json.dump(default_patterns, f, indent=2)
    print(f"  Created: {patterns_file}")
    
    # Write test files
    for filename, content in default_files.items():
        filepath = test_dir / filename
        filepath.write_text(content)
        print(f"  Created: {filepath}")
    
    print(f"\n{Colors.GREEN}Initialized {len(default_patterns)} patterns{Colors.RESET}")
    print(f"Location: {test_dir.absolute()}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI for input pattern discovery."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Input Pattern Discovery")
    parser.add_argument("program", nargs="?", help="Program file to analyze")
    parser.add_argument("--init", action="store_true", help="Initialize test_inputs folder")
    parser.add_argument("--list", action="store_true", help="List all patterns")
    parser.add_argument("--dir", default="test_inputs", help="Test inputs directory")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    if args.init:
        print(f"\n{Colors.BOLD}Initializing test inputs...{Colors.RESET}\n")
        initialize_test_inputs(args.dir)
        return 0
    
    if not Path(args.dir).exists():
        print(f"{Colors.RED}Error: {args.dir} not found{Colors.RESET}")
        print(f"Run with --init to create default files")
        return 1
    
    discovery = InputDiscovery(test_inputs_dir=args.dir)
    
    if args.list:
        print(f"\n{Colors.BOLD}Available Patterns{Colors.RESET}")
        print(f"Directory: {discovery.test_inputs_dir.absolute()}\n")
        
        for name, info in discovery.patterns.items():
            print(f"{Colors.CYAN}{name}{Colors.RESET}")
            print(f"  Description: {info.get('description', 'N/A')}")
            print(f"  Probe: {info.get('probe', [])}")
            print(f"  File: {info.get('file', 'N/A')}")
            print()
        return 0
    
    if args.program:
        code = Path(args.program).read_text()
        result = discovery.discover(code, verbose=not args.quiet)
        
        if result.success:
            inputs = discovery.load_inputs(result.patterns)
            print(f"\nLoaded {len(inputs)} input file(s)")
            for name, content in inputs.items():
                lines = content.strip().split('\n')
                print(f"  {name}: {len(lines)} lines")
            return 0
        else:
            return 1
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())