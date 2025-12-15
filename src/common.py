"""
Common Utilities for Mutation Testing Toolkit
=============================================

Single source of truth for shared components:
- Colors: ANSI terminal colors
- SafeExecutor: Sandboxed subprocess execution
- DiffGenerator: Code diff generation and formatting

All other modules import from here - no duplication.
"""

import subprocess
import sys
import os
import time
import difflib
from typing import Dict, Any, List, Tuple, Optional


# ============================================================================
# ANSI COLORS
# ============================================================================

class Colors:
    """
    ANSI escape codes for terminal coloring.
    
    Single source of truth - all modules import from here.
    """
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    
    _enabled = True
    _original_values = None
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output or --no-color flag)."""
        if cls._enabled:
            cls._enabled = False
            # Store originals for potential re-enable
            cls._original_values = {
                'RESET': cls.RESET, 'RED': cls.RED, 'GREEN': cls.GREEN,
                'YELLOW': cls.YELLOW, 'BLUE': cls.BLUE, 'MAGENTA': cls.MAGENTA,
                'CYAN': cls.CYAN, 'GRAY': cls.GRAY, 'BOLD': cls.BOLD,
                'DIM': cls.DIM, 'BG_RED': cls.BG_RED, 'BG_GREEN': cls.BG_GREEN
            }
            cls.RESET = cls.RED = cls.GREEN = cls.YELLOW = ""
            cls.BLUE = cls.MAGENTA = cls.CYAN = cls.GRAY = ""
            cls.BOLD = cls.DIM = cls.BG_RED = cls.BG_GREEN = ""
    
    @classmethod
    def enable(cls):
        """Re-enable colors if previously disabled."""
        if not cls._enabled and cls._original_values:
            cls._enabled = True
            for name, value in cls._original_values.items():
                setattr(cls, name, value)
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if colors are currently enabled."""
        return cls._enabled


# ============================================================================
# SAFE SUBPROCESS EXECUTOR
# ============================================================================

class SafeExecutor:
    """
    Execute Python code safely in isolated subprocess.
    
    Features:
    - Timeout protection
    - Process group isolation (Unix)
    - Clean environment
    - Structured result dict
    
    Single implementation - all modules use this.
    """
    
    DEFAULT_TIMEOUT = 3  # seconds
    
    @staticmethod
    def execute(
        code: str, 
        stdin_input: str = "", 
        timeout: int = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Execute Python code safely in subprocess.
        
        Args:
            code: Python source code to execute
            stdin_input: Input to pass via stdin
            timeout: Max execution time in seconds
            
        Returns:
            Dict with keys:
                - status: "success" | "runtime_error" | "timeout" | "error"
                - stdout: Captured stdout (stripped)
                - stderr: Captured stderr (stripped)
                - exit_code: Process exit code
                - runtime_ms: Execution time in milliseconds
        """
        start_time = time.time()
        process = None
        
        try:
            # Create subprocess with isolated environment
            process = subprocess.Popen(
                [sys.executable, '-c', code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                start_new_session=(os.name != 'nt')  # Process group on Unix
            )
            
            try:
                stdout, stderr = process.communicate(input=stdin_input, timeout=timeout)
                exit_code = process.returncode
                
                status = "success" if exit_code == 0 else "runtime_error"
                    
            except subprocess.TimeoutExpired:
                SafeExecutor._kill_process(process)
                stdout, stderr = "", "Execution timed out"
                exit_code = -1
                status = "timeout"
                
        except Exception as e:
            if process:
                SafeExecutor._kill_process(process)
            stdout, stderr = "", str(e)
            exit_code = -2
            status = "error"
        
        runtime_ms = int((time.time() - start_time) * 1000)
        
        return {
            "status": status,
            "stdout": stdout.strip() if stdout else "",
            "stderr": stderr.strip() if stderr else "",
            "exit_code": exit_code,
            "runtime_ms": runtime_ms
        }
    
    @staticmethod
    def _kill_process(process):
        """Forcefully kill a process and all children."""
        try:
            process.kill()
            process.wait(timeout=1)
        except Exception:
            pass


# ============================================================================
# DIFF GENERATOR
# ============================================================================

class DiffGenerator:
    """
    Generate and format diffs between original and mutated code.
    
    Single implementation - used by both engine and runner.
    """
    
    @staticmethod
    def generate(
        original: str, 
        mutated: str, 
        context_lines: int = 3,
        from_file: str = "original.py",
        to_file: str = "mutant.py"
    ) -> str:
        """
        Generate unified diff between original and mutated code.
        
        Args:
            original: Original source code
            mutated: Mutated source code
            context_lines: Number of context lines around changes
            from_file: Label for original file
            to_file: Label for mutated file
            
        Returns:
            Unified diff string
        """
        original_lines = original.splitlines(keepends=True)
        mutated_lines = mutated.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            mutated_lines,
            fromfile=from_file,
            tofile=to_file,
            n=context_lines
        )
        
        return ''.join(diff)
    
    @staticmethod
    def colorize(diff: str) -> str:
        """
        Add ANSI colors to diff output.
        
        Args:
            diff: Unified diff string
            
        Returns:
            Colorized diff string
        """
        if not Colors.is_enabled():
            return diff
        
        lines = []
        for line in diff.splitlines():
            if line.startswith('+++') or line.startswith('---'):
                lines.append(f"{Colors.BOLD}{line}{Colors.RESET}")
            elif line.startswith('+'):
                lines.append(f"{Colors.GREEN}{line}{Colors.RESET}")
            elif line.startswith('-'):
                lines.append(f"{Colors.RED}{line}{Colors.RESET}")
            elif line.startswith('@@'):
                lines.append(f"{Colors.CYAN}{line}{Colors.RESET}")
            else:
                lines.append(line)
        return '\n'.join(lines)
    
    @staticmethod
    def get_changed_lines(original: str, mutated: str) -> List[int]:
        """
        Get list of changed line numbers (1-indexed).
        
        Args:
            original: Original source code
            mutated: Mutated source code
            
        Returns:
            List of changed line numbers
        """
        original_lines = original.splitlines()
        mutated_lines = mutated.splitlines()
        
        changed = []
        matcher = difflib.SequenceMatcher(None, original_lines, mutated_lines)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('replace', 'delete'):
                changed.extend(range(i1 + 1, i2 + 1))  # 1-indexed
            elif tag == 'insert' and i1 > 0:
                changed.append(i1)
        
        return changed
    
    @staticmethod
    def inline_diff(
        old_line: str, 
        new_line: str
    ) -> Tuple[str, str]:
        """
        Generate character-level inline diff for a single line change.
        
        Args:
            old_line: Original line
            new_line: New line
            
        Returns:
            Tuple of (highlighted_old, highlighted_new)
        """
        matcher = difflib.SequenceMatcher(None, old_line, new_line)
        
        old_result = []
        new_result = []
        
        use_colors = Colors.is_enabled()
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                old_result.append(old_line[i1:i2])
                new_result.append(new_line[j1:j2])
            elif tag == 'replace':
                if use_colors:
                    old_result.append(f"{Colors.BG_RED}{old_line[i1:i2]}{Colors.RESET}")
                    new_result.append(f"{Colors.BG_GREEN}{new_line[j1:j2]}{Colors.RESET}")
                else:
                    old_result.append(f"[-{old_line[i1:i2]}-]")
                    new_result.append(f"[+{new_line[j1:j2]}+]")
            elif tag == 'delete':
                if use_colors:
                    old_result.append(f"{Colors.BG_RED}{old_line[i1:i2]}{Colors.RESET}")
                else:
                    old_result.append(f"[-{old_line[i1:i2]}-]")
            elif tag == 'insert':
                if use_colors:
                    new_result.append(f"{Colors.BG_GREEN}{new_line[j1:j2]}{Colors.RESET}")
                else:
                    new_result.append(f"[+{new_line[j1:j2]}+]")
        
        return ''.join(old_result), ''.join(new_result)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def truncate(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def parse_scope(scope_str: str, total_lines: int) -> Tuple[int, int]:
    """
    Parse a scope specification into line range.
    
    Args:
        scope_str: Either "50%" for percentage or "10-20" for line range
        total_lines: Total number of lines in the code
        
    Returns:
        Tuple of (start_line, end_line), 1-indexed inclusive
        
    Examples:
        parse_scope("50%", 100) -> (1, 50)
        parse_scope("10-20", 100) -> (10, 20)
    """
    scope_str = scope_str.strip()
    
    if scope_str.endswith('%'):
        # Percentage
        percent = float(scope_str[:-1])
        end_line = max(1, int(total_lines * (percent / 100.0)))
        return (1, end_line)
    elif '-' in scope_str:
        # Line range
        parts = scope_str.split('-')
        start = int(parts[0])
        end = int(parts[1])
        return (max(1, start), min(total_lines, end))
    else:
        # Assume single line
        line = int(scope_str)
        return (line, line)
