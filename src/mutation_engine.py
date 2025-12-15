"""
Mutation Engine
===============

AST-based mutation engine for generating code mutants.

Features:
- 56 mutation patterns (semantic-preserving and behavior-changing)
- Location tracking for each mutation
- Higher-order mutation support (combining multiple mutations)
- Line range and coverage filtering

This module handles GENERATION only. Execution is handled by mutation_runner.py.
"""

import ast
import random
import copy
import itertools
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

# Import shared utilities - single source of truth
from common import Colors, DiffGenerator


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MutationLocation:
    """Precise location of a mutation in source code."""
    line: int                    # 1-indexed line number
    column: int                  # 0-indexed column offset
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    
    def __str__(self):
        if self.end_line and self.end_line != self.line:
            return f"L{self.line}:{self.column}-L{self.end_line}:{self.end_column}"
        return f"L{self.line}:{self.column}"
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to (line, column) tuple."""
        return (self.line, self.column)


@dataclass 
class MutationResult:
    """Result of a single mutation attempt."""
    original_code: str
    mutated_code: str
    pattern_name: str
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    location: Optional[MutationLocation] = None
    mutation_id: Optional[str] = None
    change_description: Optional[str] = None  # e.g., "a + b → a - b"
    
    def get_diff(self, context_lines: int = 3) -> str:
        """Get unified diff between original and mutated code."""
        return DiffGenerator.generate(
            self.original_code, 
            self.mutated_code, 
            context_lines=context_lines
        )
    
    def get_summary(self) -> str:
        """Get a one-line summary of the mutation."""
        loc = f" at {self.location}" if self.location else ""
        status = "✓" if self.is_valid else "✗"
        return f"[{status}] {self.pattern_name}{loc}"


@dataclass
class CombinedMutationResult:
    """Result of applying multiple mutations (higher-order)."""
    original_code: str
    mutated_code: str
    applied_mutations: List[int]
    mutation_names: List[str]
    is_valid: bool
    locations: List['MutationLocation'] = field(default_factory=list)  # Track ALL mutation locations
    conflicts: List[str] = field(default_factory=list)
    skipped_mutations: List[int] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    
    def get_diff(self, context_lines: int = 3) -> str:
        """Get unified diff between original and mutated code."""
        return DiffGenerator.generate(
            self.original_code, 
            self.mutated_code, 
            context_lines=context_lines
        )
    
    def get_locations_str(self) -> str:
        """Get formatted string of all mutation locations."""
        if not self.locations:
            return "unknown"
        return ", ".join(f"L{loc.line}" for loc in self.locations)


@dataclass
class TemplateMutation:
    """Definition of a mutation pattern."""
    name: str
    precondition_checker: callable
    transformer: callable


# ============================================================================
# MUTATION CATEGORIES
# ============================================================================

# Behavior-changing mutations - these actually alter program semantics
BEHAVIOR_CHANGING = [
    13, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55
]

# Semantic-preserving mutations - refactoring that shouldn't change behavior
SEMANTIC_PRESERVING = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22
]

BEHAVIOR_CHANGING_SET = set(BEHAVIOR_CHANGING)
SEMANTIC_PRESERVING_SET = set(SEMANTIC_PRESERVING)


# ============================================================================
# MUTATION ENGINE
# ============================================================================

class MutationEngine:
    """
    AST-based mutation engine.
    
    Generates mutants by applying transformation patterns to source code.
    
    Usage:
        engine = MutationEngine()
        results = engine.mutate(source_code)
        
        # Filter to behavior-changing only
        results = engine.mutate(source_code, mutations=BEHAVIOR_CHANGING)
        
        # Limit to specific lines
        results = engine.mutate(source_code, line_range=(10, 20))
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.stats = {"attempted": 0, "success": 0}
        self._mutation_counter = 0
    
    def _next_mutation_id(self) -> str:
        """Generate unique mutation ID."""
        self._mutation_counter += 1
        return f"mutant_{self._mutation_counter:04d}"
    
    def reset_counter(self):
        """Reset mutation ID counter."""
        self._mutation_counter = 0
    
    def list_mutations(self) -> List[str]:
        """List all available mutation patterns."""
        return [t.name for t in self.templates]
    
    def get_mutation_name(self, index: int) -> Optional[str]:
        """Get mutation name by index."""
        if 0 <= index < len(self.templates):
            return self.templates[index].name
        return None
    
    def get_mutation_index(self, name: str) -> Optional[int]:
        """Get mutation index by name."""
        for i, t in enumerate(self.templates):
            if t.name == name:
                return i
        return None
    
    def get_applicable(self, code: str) -> List[int]:
        """Get indices of mutations applicable to this code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        return [i for i, t in enumerate(self.templates) if t.precondition_checker(tree)]
    
    def mutate(
        self, 
        code: str, 
        mutations: Optional[List[int]] = None,
        line_range: Optional[Tuple[int, int]] = None,
        behavior_only: bool = False
    ) -> List[MutationResult]:
        """
        Apply mutations to code.
        
        Args:
            code: Python source code to mutate
            mutations: List of mutation indices to apply, or None for all applicable
            line_range: Optional (start_line, end_line) tuple to limit mutations (1-indexed)
            behavior_only: If True, only apply behavior-changing mutations
            
        Returns:
            List of MutationResult objects
        """
        total_lines = len(code.splitlines())
        
        # Determine effective line range
        if line_range:
            effective_start, effective_end = line_range
        else:
            effective_start, effective_end = 1, total_lines
        
        # Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [MutationResult(
                original_code=code,
                mutated_code=code,
                pattern_name="parse_error",
                is_valid=False,
                validation_errors=[f"Failed to parse: {e}"]
            )]
        
        # Determine which mutations to apply
        if mutations is not None:
            indices = [i for i in mutations if 0 <= i < len(self.templates)]
        elif behavior_only:
            indices = [i for i in BEHAVIOR_CHANGING if self.templates[i].precondition_checker(tree)]
        else:
            indices = [i for i, t in enumerate(self.templates) if t.precondition_checker(tree)]
        
        results = []
        for idx in indices:
            template = self.templates[idx]
            
            if not template.precondition_checker(tree):
                continue
            
            self.stats["attempted"] += 1
            
            try:
                # Call the transformer
                transformer_result = template.transformer(code, tree)
                
                # Check if this returns a list (multiple mutations) or tuple (single mutation)
                if isinstance(transformer_result, list):
                    # Multi-mutation transformer
                    mutation_pairs = transformer_result
                else:
                    # Single-mutation transformer - wrap in list
                    mutation_pairs = [transformer_result]
                
                # Process each mutation
                for mutated_code, location in mutation_pairs:
                    # Check if mutation is within allowed line range
                    if location is not None:
                        if not (effective_start <= location.line <= effective_end):
                            continue  # Skip mutations outside range
                    
                    # Check if actually changed
                    if mutated_code.strip() == code.strip():
                        continue
                    
                    # Validate syntax
                    try:
                        ast.parse(mutated_code)
                        self.stats["success"] += 1
                        
                        # Infer location from diff if not provided
                        if location is None:
                            changed = DiffGenerator.get_changed_lines(code, mutated_code)
                            if changed:
                                location = MutationLocation(line=changed[0], column=0)
                                # Re-check line range
                                if not (effective_start <= location.line <= effective_end):
                                    continue
                        
                        results.append(MutationResult(
                            original_code=code,
                            mutated_code=mutated_code,
                            pattern_name=template.name,
                            is_valid=True,
                            location=location,
                            mutation_id=self._next_mutation_id()
                        ))
                        
                    except SyntaxError as e:
                        results.append(MutationResult(
                            original_code=code,
                            mutated_code=mutated_code,
                            pattern_name=template.name,
                            is_valid=False,
                            validation_errors=[f"Syntax error: {e}"]
                        ))
                    
            except Exception as e:
                # Transformer failed - skip silently
                pass
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mutation statistics."""
        rate = self.stats["success"] / self.stats["attempted"] if self.stats["attempted"] > 0 else 0
        return {**self.stats, "success_rate": rate}
    
    def _unparse(self, tree: ast.AST) -> str:
        """Safely unparse AST tree to source code."""
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    
    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================
    
    def format_mutation(self, result: MutationResult, context_lines: int = 3) -> str:
        """
        Format a mutation result for display.
        
        Args:
            result: MutationResult to format
            context_lines: Lines of context around changes
            
        Returns:
            Formatted string for terminal output
        """
        lines = []
        c = Colors
        
        # Header
        loc_str = f" at {result.location}" if result.location else ""
        mid = result.mutation_id or "unnamed"
        lines.append(f"{c.BOLD}┌─ {mid}: {result.pattern_name}{loc_str}{c.RESET}")
        lines.append("│")
        
        # Show diff
        diff = result.get_diff(context_lines)
        if diff:
            for line in DiffGenerator.colorize(diff).splitlines()[:20]:
                lines.append(f"│  {line}")
        else:
            lines.append(f"│  {c.DIM}(no diff available){c.RESET}")
        
        lines.append("│")
        
        # Status
        if result.is_valid:
            status = f"{c.GREEN}✓ Valid{c.RESET}"
        else:
            errors = ', '.join(result.validation_errors)
            status = f"{c.RED}✗ Invalid: {errors}{c.RESET}"
        lines.append(f"└─ Status: {status}")
        
        return '\n'.join(lines)
    
    # =========================================================================
    # TEMPLATE INITIALIZATION
    # =========================================================================
    
    def _initialize_templates(self) -> List[TemplateMutation]:
        """Initialize all mutation patterns."""
        return [
            # 0-2: Basic mutations
            TemplateMutation("variable_renaming", self._has_variables, self._rename_variable),
            TemplateMutation("dead_code_insertion", lambda _: True, self._insert_dead_code),
            TemplateMutation("boolean_rewriting", self._has_boolean_expr, self._rewrite_boolean),
            
            # 3-11: Arithmetic identities
            TemplateMutation("arithmetic_add_zero", self._has_arithmetic, self._make_arithmetic_identity(0)),
            TemplateMutation("arithmetic_zero_add", self._has_arithmetic, self._make_arithmetic_identity(1)),
            TemplateMutation("arithmetic_sub_zero", self._has_arithmetic, self._make_arithmetic_identity(2)),
            TemplateMutation("arithmetic_mul_one", self._has_arithmetic, self._make_arithmetic_identity(3)),
            TemplateMutation("arithmetic_one_mul", self._has_arithmetic, self._make_arithmetic_identity(4)),
            TemplateMutation("arithmetic_div_one", self._has_arithmetic, self._make_arithmetic_identity(5)),
            TemplateMutation("arithmetic_pow_one", self._has_arithmetic, self._make_arithmetic_identity(6)),
            TemplateMutation("arithmetic_double_neg", self._has_arithmetic, self._make_arithmetic_identity(7)),
            TemplateMutation("arithmetic_unary_pos", self._has_arithmetic, self._make_arithmetic_identity(8)),
            
            # 12-22: Refactoring mutations
            TemplateMutation("constant_extraction", self._has_literals, self._extract_constant),
            TemplateMutation("if_else_negation", self._has_if_else, self._negate_if_else),
            TemplateMutation("comment_insertion", lambda _: True, self._add_comment),
            TemplateMutation("for_to_while", self._has_for_range, self._for_to_while),
            TemplateMutation("list_comp_to_loop", self._has_list_comp, self._list_comp_to_loop),
            TemplateMutation("lambda_to_def", self._has_lambda, self._lambda_to_def),
            TemplateMutation("early_return_elimination", self._has_early_return, self._eliminate_early_return),
            TemplateMutation("swap_binary_operands", self._has_commutative_binop, self._swap_binary_operands),
            TemplateMutation("inline_variable", self._has_single_use_var, self._inline_variable),
            TemplateMutation("extract_variable", self._has_complex_expr, self._extract_variable),
            TemplateMutation("loop_unrolling", self._has_small_range_loop, self._unroll_loop),
            
            # 23-29: Core semantic-changing mutations
            TemplateMutation("off_by_one", self._has_for_range, self._off_by_one),
            TemplateMutation("arithmetic_op_replacement", self._has_binop, self._replace_arithmetic_op),
            TemplateMutation("logical_op_replacement", self._has_logical_op, self._replace_logical_op),
            TemplateMutation("remove_function_call", self._has_removable_call, self._remove_function_call),
            TemplateMutation("statement_deletion", self._has_deletable_stmt, self._delete_statement),
            TemplateMutation("relational_op_replacement", self._has_relational_op, self._replace_relational_op),
            TemplateMutation("relational_op_all", self._has_relational_op, self._replace_relational_op_all),
            
            # 30-55: Additional mutmut-style mutations
            TemplateMutation("power_op_replacement", self._has_power_op, self._replace_power_op),
            TemplateMutation("bitshift_op_replacement", self._has_bitshift_op, self._replace_bitshift_op),
            TemplateMutation("bitwise_op_replacement", self._has_bitwise_op, self._replace_bitwise_op),
            TemplateMutation("remove_not", self._has_not_op, self._remove_not),
            TemplateMutation("negate_unary", self._has_unary_op, self._negate_unary),
            TemplateMutation("is_op_replacement", self._has_is_op, self._replace_is_op),
            TemplateMutation("in_op_replacement", self._has_in_op, self._replace_in_op),
            TemplateMutation("bool_constant_swap", self._has_bool_constant, self._swap_bool_constant),
            TemplateMutation("zero_one_swap", self._has_zero_or_one, self._swap_zero_one),
            TemplateMutation("empty_list_mutation", self._has_empty_list, self._mutate_empty_list),
            TemplateMutation("empty_dict_mutation", self._has_empty_dict, self._mutate_empty_dict),
            TemplateMutation("empty_tuple_mutation", self._has_empty_tuple, self._mutate_empty_tuple),
            TemplateMutation("if_true_to_false", self._has_if_stmt, self._if_cond_to_false),
            TemplateMutation("remove_if_body", self._has_if_stmt, self._remove_if_body),
            TemplateMutation("remove_else_body", self._has_else_block, self._remove_else_body),
            TemplateMutation("while_to_false", self._has_while_stmt, self._while_cond_to_false),
            TemplateMutation("for_to_empty", self._has_for_stmt, self._for_to_empty_iter),
            TemplateMutation("return_to_none", self._has_return_value, self._return_to_none),
            TemplateMutation("return_none_to_one", self._has_return_none, self._return_none_to_one),
            TemplateMutation("remove_raise", self._has_raise, self._remove_raise),
            TemplateMutation("remove_except_body", self._has_try_except, self._remove_except_body),
            TemplateMutation("remove_break", self._has_break, self._remove_break),
            TemplateMutation("remove_continue", self._has_continue, self._remove_continue),
            TemplateMutation("len_check_to_false", self._has_len_check, self._len_check_to_false),
            TemplateMutation("len_eq_zero_to_true", self._has_len_eq_zero, self._len_eq_zero_to_true),
            TemplateMutation("len_gt_zero_to_false", self._has_len_gt_zero, self._len_gt_zero_to_false),
        ]
    
    def _make_arithmetic_identity(self, identity_type: int):
        """Factory for arithmetic identity transformers."""
        def transformer(code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
            return self._apply_arithmetic_identity(code, tree, identity_type)
        return transformer
    
    # =========================================================================
    # PRECONDITION CHECKERS
    # =========================================================================
    
    def _has_variables(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                return True
        return False
    
    def _has_boolean_expr(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.BoolOp) for n in ast.walk(tree))
    
    def _has_arithmetic(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.value, (ast.Name, ast.Constant)):
                    return True
        return False
    
    def _has_literals(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in [0, 1, -1]:
                    return True
        return False
    
    def _has_if_else(self, tree: ast.AST) -> bool:
        all_ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
        elif_nodes = set()
        for node in all_ifs:
            for item in node.orelse:
                if isinstance(item, ast.If):
                    elif_nodes.add(id(item))
        for node in all_ifs:
            if id(node) not in elif_nodes and node.orelse:
                if not any(isinstance(item, ast.If) for item in node.orelse):
                    return True
        return False
    
    def _has_for_range(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                    if 1 <= len(node.iter.args) <= 3:
                        return True
        return False
    
    def _has_list_comp(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.ListComp):
                if len(node.targets) == 1:
                    comp = node.value
                    if len(comp.generators) == 1 and not comp.generators[0].ifs:
                        return True
        return False
    
    def _has_lambda(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    return True
        return False
    
    def _has_early_return(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and len(node.body) >= 2:
                if not isinstance(node.body[-1], ast.Return):
                    continue
                for stmt in node.body[:-1]:
                    if isinstance(stmt, ast.If):
                        if (len(stmt.body) == 1 and 
                            isinstance(stmt.body[0], ast.Return) and
                            not stmt.orelse):
                            return True
        return False
    
    def _has_commutative_binop(self, tree: ast.AST) -> bool:
        commutative_ops = (ast.Add, ast.Mult, ast.Eq, ast.NotEq, ast.BitAnd, ast.BitOr, ast.BitXor)
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, commutative_ops):
                return True
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                if isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
                    return True
        return False
    
    def _has_single_use_var(self, tree: ast.AST) -> bool:
        definition_counts = {}
        usages = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    definition_counts[var_name] = definition_counts.get(var_name, 0) + 1
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                usages[node.id] = usages.get(node.id, 0) + 1
        for var_name, def_count in definition_counts.items():
            if def_count == 1 and usages.get(var_name, 0) == 1:
                return True
        return False
    
    def _has_complex_expr(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.left, ast.BinOp) or isinstance(node.right, ast.BinOp):
                    return True
            if isinstance(node, ast.Call) and node.args:
                return True
        return False
    
    def _has_small_range_loop(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                    if len(node.iter.args) == 1:
                        arg = node.iter.args[0]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                            if 2 <= arg.value <= 4:
                                return True
        return False
    
    def _has_binop(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.BinOp) for n in ast.walk(tree))
    
    def _has_logical_op(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.BoolOp):
                return True
            if isinstance(node, ast.Compare):
                return True
        return False
    
    def _has_removable_call(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                return True
        return False
    
    def _has_deletable_stmt(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef)):
                if len(node.body) >= 2:
                    return True
        return False
    
    def _has_relational_op(self, tree: ast.AST) -> bool:
        relational_ops = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, relational_ops):
                        return True
        return False
    
    def _has_power_op(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) for n in ast.walk(tree))
    
    def _has_bitshift_op(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.BinOp) and isinstance(n.op, (ast.LShift, ast.RShift)) for n in ast.walk(tree))
    
    def _has_bitwise_op(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.BinOp) and isinstance(n.op, (ast.BitAnd, ast.BitOr, ast.BitXor)) for n in ast.walk(tree))
    
    def _has_not_op(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) for n in ast.walk(tree))
    
    def _has_unary_op(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)) for n in ast.walk(tree))
    
    def _has_is_op(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, (ast.Is, ast.IsNot)):
                        return True
        return False
    
    def _has_in_op(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, (ast.In, ast.NotIn)):
                        return True
        return False
    
    def _has_bool_constant(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, bool):
                return True
        return False
    
    def _has_zero_or_one(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in (0, 1):
                if isinstance(node.value, int) and not isinstance(node.value, bool):
                    return True
        return False
    
    def _has_empty_list(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.List) and len(n.elts) == 0 for n in ast.walk(tree))
    
    def _has_empty_dict(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.Dict) and len(n.keys) == 0 for n in ast.walk(tree))
    
    def _has_empty_tuple(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.Tuple) and len(n.elts) == 0 for n in ast.walk(tree))
    
    def _has_if_stmt(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.If) for n in ast.walk(tree))
    
    def _has_else_block(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and node.orelse:
                if not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)):
                    return True
        return False
    
    def _has_while_stmt(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.While) for n in ast.walk(tree))
    
    def _has_for_stmt(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.For) for n in ast.walk(tree))
    
    def _has_return_value(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value is not None:
                if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                    return True
        return False
    
    def _has_return_none(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if node.value is None:
                    return True
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    return True
        return False
    
    def _has_raise(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.Raise) for n in ast.walk(tree))
    
    def _has_try_except(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.Try) and n.handlers for n in ast.walk(tree))
    
    def _has_break(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.Break) for n in ast.walk(tree))
    
    def _has_continue(self, tree: ast.AST) -> bool:
        return any(isinstance(n, ast.Continue) for n in ast.walk(tree))
    
    def _has_len_check(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Call):
                    if isinstance(test.func, ast.Name) and test.func.id == 'len':
                        return True
        return False
    
    def _has_len_eq_zero(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                if isinstance(node.ops[0], ast.Eq):
                    if isinstance(node.left, ast.Call):
                        if isinstance(node.left.func, ast.Name) and node.left.func.id == 'len':
                            if len(node.comparators) == 1:
                                comp = node.comparators[0]
                                if isinstance(comp, ast.Constant) and comp.value == 0:
                                    return True
        return False
    
    def _has_len_gt_zero(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                if isinstance(node.ops[0], ast.Gt):
                    if isinstance(node.left, ast.Call):
                        if isinstance(node.left.func, ast.Name) and node.left.func.id == 'len':
                            if len(node.comparators) == 1:
                                comp = node.comparators[0]
                                if isinstance(comp, ast.Constant) and comp.value == 0:
                                    return True
        return False
    
    # =========================================================================
    # TRANSFORMERS
    # =========================================================================
    
    def _rename_variable(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Rename a random variable."""
        class Collector(ast.NodeVisitor):
            def __init__(self):
                self.vars = []
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.vars.append((node.id, node.lineno, node.col_offset))
                self.generic_visit(node)
        
        collector = Collector()
        collector.visit(tree)
        
        builtins = {'print', 'len', 'range', 'str', 'int', 'list', 'dict', 'set', 'tuple'}
        renamable = [(v, ln, co) for v, ln, co in collector.vars 
                     if not v.startswith('_') and v not in builtins]
        
        if not renamable:
            return code, None
        
        old, line, col = random.choice(renamable)
        new = f"var_{random.randint(1000, 9999)}"
        location = MutationLocation(line=line, column=col)
        
        class Renamer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == old:
                    node.id = new
                return node
        
        new_tree = Renamer().visit(copy.deepcopy(tree))
        return self._unparse(new_tree), location
    
    def _insert_dead_code(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Insert dead code (if False: pass)."""
        class Inserter(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            def visit_Module(self, node):
                if not self.done and node.body:
                    dead = ast.If(test=ast.Constant(False), body=[ast.Pass()], orelse=[])
                    pos = random.randint(0, len(node.body))
                    node.body.insert(pos, dead)
                    self.done = True
                    if node.body and hasattr(node.body[0], 'lineno'):
                        self.location = MutationLocation(line=node.body[0].lineno, column=0)
                    else:
                        self.location = MutationLocation(line=1, column=0)
                return self.generic_visit(node)
        
        inserter = Inserter()
        new_tree = inserter.visit(copy.deepcopy(tree))
        return self._unparse(new_tree), inserter.location
    
    def _rewrite_boolean(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Apply De Morgan's law."""
        class Rewriter(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            def visit_BoolOp(self, node):
                if not self.done:
                    not_vals = [ast.UnaryOp(op=ast.Not(), operand=v) for v in node.values]
                    if isinstance(node.op, ast.And):
                        inner = ast.BoolOp(op=ast.Or(), values=not_vals)
                    else:
                        inner = ast.BoolOp(op=ast.And(), values=not_vals)
                    self.done = True
                    self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                    return ast.UnaryOp(op=ast.Not(), operand=inner)
                return self.generic_visit(node)
        
        rewriter = Rewriter()
        new_tree = rewriter.visit(copy.deepcopy(tree))
        if not rewriter.done:
            return code, None
        return self._unparse(new_tree), rewriter.location
    
    def _apply_arithmetic_identity(self, code: str, tree: ast.AST, 
                                   identity_type: int) -> Tuple[str, Optional[MutationLocation]]:
        """Apply arithmetic identity transformation."""
        class Adder(ast.NodeTransformer):
            def __init__(self, id_type: int):
                self.done = False
                self.identity_type = id_type
                self.location = None
            
            def _apply_identity(self, value_node):
                t = self.identity_type
                if t == 0:
                    return ast.BinOp(left=value_node, op=ast.Add(), right=ast.Constant(0))
                elif t == 1:
                    return ast.BinOp(left=ast.Constant(0), op=ast.Add(), right=value_node)
                elif t == 2:
                    return ast.BinOp(left=value_node, op=ast.Sub(), right=ast.Constant(0))
                elif t == 3:
                    return ast.BinOp(left=value_node, op=ast.Mult(), right=ast.Constant(1))
                elif t == 4:
                    return ast.BinOp(left=ast.Constant(1), op=ast.Mult(), right=value_node)
                elif t == 5:
                    return ast.BinOp(left=value_node, op=ast.Div(), right=ast.Constant(1))
                elif t == 6:
                    return ast.BinOp(left=value_node, op=ast.Pow(), right=ast.Constant(1))
                elif t == 7:
                    return ast.UnaryOp(op=ast.USub(), 
                                       operand=ast.UnaryOp(op=ast.USub(), operand=value_node))
                elif t == 8:
                    return ast.UnaryOp(op=ast.UAdd(), operand=value_node)
                return value_node
            
            def visit_Assign(self, node):
                if not self.done and len(node.targets) == 1:
                    if isinstance(node.value, (ast.Name, ast.Constant)):
                        node.value = self._apply_identity(node.value)
                        self.done = True
                        self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                return node
        
        adder = Adder(identity_type)
        new_tree = adder.visit(copy.deepcopy(tree))
        return self._unparse(new_tree), adder.location

    def _extract_constant(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Extract a numeric literal to a constant."""
        literals = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in [0, 1, -1]:
                    literals.append((node.value, node.lineno, node.col_offset))
        
        if not literals:
            return code, None
        
        value, line, col = random.choice(literals)
        name = f"CONST_{random.randint(1000, 9999)}"
        location = MutationLocation(line=line, column=col)
        
        class Extractor(ast.NodeTransformer):
            def __init__(self):
                self.done = False
            def visit_Constant(self, node):
                if not self.done and node.value == value:
                    self.done = True
                    return ast.Name(id=name, ctx=ast.Load())
                return node
        
        new_tree = Extractor().visit(copy.deepcopy(tree))
        const_def = ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Constant(value=value)
        )
        new_tree.body.insert(0, const_def)
        return self._unparse(new_tree), location
    
    def _negate_if_else(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Negate if condition and swap branches."""
        class Negator(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            def visit_If(self, node):
                if not self.done and node.orelse:
                    if not any(isinstance(i, ast.If) for i in node.orelse):
                        new_test = ast.UnaryOp(op=ast.Not(), operand=node.test)
                        result = ast.If(test=new_test, body=node.orelse, orelse=node.body)
                        self.done = True
                        self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                        return result
                return self.generic_visit(node)
        
        negator = Negator()
        new_tree = negator.visit(copy.deepcopy(tree))
        return self._unparse(new_tree), negator.location
    
    def _add_comment(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Add a comment to the code."""
        comments = ["# Mutation applied", "# Semantically equivalent", "# Transformed code"]
        lines = code.split('\n')
        pos = random.randint(0, len(lines))
        lines.insert(pos, random.choice(comments))
        return '\n'.join(lines), MutationLocation(line=pos + 1, column=0)
    
    def _for_to_while(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Convert for-range to while loop."""
        location = None
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                    location = MutationLocation(line=node.lineno, column=node.col_offset)
                    break
        
        class Transformer(ast.NodeTransformer):
            def __init__(self):
                self.done = False
            def visit_For(self, node):
                if self.done:
                    return node
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        args = node.iter.args
                        if not (1 <= len(args) <= 3):
                            return self.generic_visit(node)
                        var = node.target.id if isinstance(node.target, ast.Name) else 'i'
                        if len(args) == 1:
                            start, end, step = ast.Constant(0), args[0], ast.Constant(1)
                        elif len(args) == 2:
                            start, end, step = args[0], args[1], ast.Constant(1)
                        else:
                            start, end, step = args[0], args[1], args[2]
                        init = ast.Assign(targets=[ast.Name(id=var, ctx=ast.Store())], value=start)
                        test = ast.Compare(left=ast.Name(id=var, ctx=ast.Load()), 
                                          ops=[ast.Lt()], comparators=[end])
                        inc = ast.AugAssign(target=ast.Name(id=var, ctx=ast.Store()), 
                                           op=ast.Add(), value=step)
                        while_node = ast.While(test=test, body=node.body + [inc], orelse=[])
                        self.done = True
                        return [init, while_node]
                return self.generic_visit(node)
            def visit_Module(self, node):
                new_body = []
                for stmt in node.body:
                    result = self.visit(stmt)
                    if isinstance(result, list):
                        new_body.extend(result)
                    else:
                        new_body.append(result)
                node.body = new_body
                return node
            def visit_FunctionDef(self, node):
                new_body = []
                for stmt in node.body:
                    result = self.visit(stmt)
                    if isinstance(result, list):
                        new_body.extend(result)
                    else:
                        new_body.append(result)
                node.body = new_body
                return node
        
        transformer = Transformer()
        new_tree = transformer.visit(copy.deepcopy(tree))
        if not transformer.done:
            return code, None
        ast.fix_missing_locations(new_tree)
        return self._unparse(new_tree), location
    
    def _list_comp_to_loop(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Convert list comprehension to for loop."""
        location = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.ListComp):
                location = MutationLocation(line=node.lineno, column=node.col_offset)
                break
        
        class Transformer(ast.NodeTransformer):
            def __init__(self):
                self.done = False
            
            def visit_Assign(self, node):
                if self.done:
                    return node
                if isinstance(node.value, ast.ListComp) and len(node.targets) == 1:
                    comp = node.value
                    if len(comp.generators) == 1 and not comp.generators[0].ifs:
                        gen = comp.generators[0]
                        target = node.targets[0]
                        init = ast.Assign(targets=[target], value=ast.List(elts=[], ctx=ast.Load()))
                        append = ast.Expr(value=ast.Call(
                            func=ast.Attribute(value=target, attr='append', ctx=ast.Load()),
                            args=[comp.elt],
                            keywords=[]
                        ))
                        loop = ast.For(target=gen.target, iter=gen.iter, body=[append], orelse=[])
                        self.done = True
                        return [init, loop]
                return node
            
            def visit_Module(self, node):
                new_body = []
                for stmt in node.body:
                    result = self.visit(stmt)
                    if isinstance(result, list):
                        new_body.extend(result)
                    else:
                        new_body.append(result)
                node.body = new_body
                return node
        
        transformer = Transformer()
        new_tree = transformer.visit(copy.deepcopy(tree))
        if not transformer.done:
            return code, None
        ast.fix_missing_locations(new_tree)
        return self._unparse(new_tree), location
    
    def _lambda_to_def(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Convert lambda to function definition."""
        location = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
                location = MutationLocation(line=node.lineno, column=node.col_offset)
                break
        
        class Transformer(ast.NodeTransformer):
            def __init__(self):
                self.done = False
            
            def visit_Assign(self, node):
                if self.done:
                    return node
                if isinstance(node.value, ast.Lambda) and len(node.targets) == 1:
                    if isinstance(node.targets[0], ast.Name):
                        name = node.targets[0].id
                        lam = node.value
                        func = ast.FunctionDef(
                            name=name,
                            args=lam.args,
                            body=[ast.Return(value=lam.body)],
                            decorator_list=[],
                            returns=None
                        )
                        self.done = True
                        return func
                return node
        
        transformer = Transformer()
        new_tree = transformer.visit(copy.deepcopy(tree))
        if not transformer.done:
            return code, None
        return self._unparse(new_tree), location
    
    def _eliminate_early_return(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Eliminate early return pattern."""
        class Transformer(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            
            def visit_FunctionDef(self, node):
                if self.done or len(node.body) < 2:
                    return node
                
                if not isinstance(node.body[-1], ast.Return):
                    return node
                
                early_return_idx = None
                for i, stmt in enumerate(node.body[:-1]):
                    if isinstance(stmt, ast.If):
                        if (len(stmt.body) == 1 and 
                            isinstance(stmt.body[0], ast.Return) and
                            not stmt.orelse):
                            early_return_idx = i
                            self.location = MutationLocation(line=stmt.lineno, column=stmt.col_offset)
                            break
                
                if early_return_idx is None:
                    return node
                
                if_node = node.body[early_return_idx]
                early_return_value = if_node.body[0].value
                code_before = node.body[:early_return_idx]
                code_after = node.body[early_return_idx + 1:]
                
                var = 'result'
                
                def transform_returns(stmts):
                    new_stmts = []
                    for stmt in stmts:
                        if isinstance(stmt, ast.Return):
                            new_stmts.append(ast.Assign(
                                targets=[ast.Name(id=var, ctx=ast.Store())],
                                value=stmt.value if stmt.value else ast.Constant(None)
                            ))
                        elif isinstance(stmt, ast.If):
                            new_stmt = copy.deepcopy(stmt)
                            new_stmt.body = transform_returns(stmt.body)
                            if stmt.orelse:
                                new_stmt.orelse = transform_returns(stmt.orelse)
                            new_stmts.append(new_stmt)
                        else:
                            new_stmts.append(copy.deepcopy(stmt))
                    return new_stmts
                
                init = ast.Assign(
                    targets=[ast.Name(id=var, ctx=ast.Store())],
                    value=ast.Constant(None)
                )
                
                if_assign = ast.Assign(
                    targets=[ast.Name(id=var, ctx=ast.Store())],
                    value=early_return_value if early_return_value else ast.Constant(None)
                )
                
                else_body = transform_returns(code_after)
                if not else_body:
                    else_body = [ast.Pass()]
                
                new_if = ast.If(
                    test=if_node.test,
                    body=[if_assign],
                    orelse=else_body
                )
                
                new_ret = ast.Return(value=ast.Name(id=var, ctx=ast.Load()))
                node.body = code_before + [init, new_if, new_ret]
                self.done = True
                
                return node
        
        transformer = Transformer()
        new_tree = transformer.visit(copy.deepcopy(tree))
        if not transformer.done:
            return code, None
        return self._unparse(new_tree), transformer.location
    
    def _swap_binary_operands(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Swap binary operands."""
        class Swapper(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            def visit_BinOp(self, node):
                if not self.done:
                    commutative = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
                    if isinstance(node.op, commutative):
                        node.left, node.right = node.right, node.left
                        self.done = True
                        self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                return self.generic_visit(node)
        
        swapper = Swapper()
        new_tree = swapper.visit(copy.deepcopy(tree))
        if not swapper.done:
            return code, None
        return self._unparse(new_tree), swapper.location
    
    def _inline_variable(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Inline a single-use variable."""
        definition_counts = {}
        definitions = {}
        usages = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    definition_counts[var_name] = definition_counts.get(var_name, 0) + 1
                    if var_name not in definitions:
                        definitions[var_name] = (node.value, node.lineno, node.col_offset)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                usages[node.id] = usages.get(node.id, 0) + 1
        
        target_var = None
        target_value = None
        location = None
        for var_name, (value, line, col) in definitions.items():
            if definition_counts.get(var_name, 0) == 1 and usages.get(var_name, 0) == 1:
                target_var = var_name
                target_value = value
                location = MutationLocation(line=line, column=col)
                break
        
        if not target_var:
            return code, None
        
        class Inliner(ast.NodeTransformer):
            def __init__(self, var_name, value):
                self.var_name = var_name
                self.value = value
                self.inlined = False
                self.removed_def = False
            
            def visit_Name(self, node):
                if not self.inlined and node.id == self.var_name and isinstance(node.ctx, ast.Load):
                    self.inlined = True
                    return copy.deepcopy(self.value)
                return node
            
            def visit_Assign(self, node):
                if not self.removed_def and len(node.targets) == 1:
                    if isinstance(node.targets[0], ast.Name) and node.targets[0].id == self.var_name:
                        self.removed_def = True
                        return None
                return self.generic_visit(node)
            
            def _filter_body(self, body):
                result = [self.visit(s) for s in body]
                result = [s for s in result if s is not None]
                if not result:
                    result = [ast.Pass()]
                return result
            
            def visit_Module(self, node):
                node.body = self._filter_body(node.body)
                return node
            
            def visit_FunctionDef(self, node):
                node.body = self._filter_body(node.body)
                return self.generic_visit(node)
        
        new_tree = Inliner(target_var, target_value).visit(copy.deepcopy(tree))
        return self._unparse(new_tree), location
    
    def _extract_variable(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Extract a subexpression to a temporary variable."""
        class Extractor(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.extracted_expr = None
                self.temp_name = f"temp_{random.randint(1000, 9999)}"
                self.location = None
            
            def visit_BinOp(self, node):
                if not self.done:
                    if isinstance(node.left, ast.BinOp):
                        self.extracted_expr = node.left
                        self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                        node.left = ast.Name(id=self.temp_name, ctx=ast.Load())
                        self.done = True
                        return node
                    elif isinstance(node.right, ast.BinOp):
                        self.extracted_expr = node.right
                        self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                        node.right = ast.Name(id=self.temp_name, ctx=ast.Load())
                        self.done = True
                        return node
                return self.generic_visit(node)
            
            def visit_Module(self, node):
                new_body = []
                for stmt in node.body:
                    new_stmt = self.visit(stmt)
                    if self.extracted_expr is not None and self.done:
                        assign = ast.Assign(
                            targets=[ast.Name(id=self.temp_name, ctx=ast.Store())],
                            value=self.extracted_expr
                        )
                        new_body.append(assign)
                        self.extracted_expr = None
                    new_body.append(new_stmt)
                node.body = new_body
                return node
        
        extractor = Extractor()
        new_tree = extractor.visit(copy.deepcopy(tree))
        if not extractor.done:
            return code, None
        return self._unparse(new_tree), extractor.location
    
    def _unroll_loop(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Unroll small range loops."""
        class Unroller(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            
            def visit_For(self, node):
                if self.done:
                    return node
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        if len(node.iter.args) == 1:
                            arg = node.iter.args[0]
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                                if 2 <= arg.value <= 4:
                                    var_name = node.target.id if isinstance(node.target, ast.Name) else 'i'
                                    self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                                    unrolled = []
                                    for i in range(arg.value):
                                        assign = ast.Assign(
                                            targets=[ast.Name(id=var_name, ctx=ast.Store())],
                                            value=ast.Constant(i)
                                        )
                                        unrolled.append(assign)
                                        for stmt in node.body:
                                            unrolled.append(copy.deepcopy(stmt))
                                    self.done = True
                                    return unrolled
                return self.generic_visit(node)
            
            def visit_Module(self, node):
                new_body = []
                for stmt in node.body:
                    result = self.visit(stmt)
                    if isinstance(result, list):
                        new_body.extend(result)
                    else:
                        new_body.append(result)
                node.body = new_body
                return node
            
            def visit_FunctionDef(self, node):
                new_body = []
                for stmt in node.body:
                    result = self.visit(stmt)
                    if isinstance(result, list):
                        new_body.extend(result)
                    else:
                        new_body.append(result)
                node.body = new_body
                return node
        
        unroller = Unroller()
        new_tree = unroller.visit(copy.deepcopy(tree))
        if not unroller.done:
            return code, None
        return self._unparse(new_tree), unroller.location
    
    def _off_by_one(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Off-by-one mutation."""
        class OffByOne(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            def visit_Call(self, node):
                if not self.done:
                    if isinstance(node.func, ast.Name) and node.func.id == 'range':
                        if node.args:
                            idx = len(node.args) - 1
                            if len(node.args) <= 2:
                                idx = 0 if len(node.args) == 1 else 1
                            original = node.args[idx]
                            op = random.choice([ast.Add(), ast.Sub()])
                            node.args[idx] = ast.BinOp(left=original, op=op, right=ast.Constant(1))
                            self.done = True
                            self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                return self.generic_visit(node)
        
        mutator = OffByOne()
        new_tree = mutator.visit(copy.deepcopy(tree))
        return self._unparse(new_tree), mutator.location
    
    def _replace_arithmetic_op(self, code: str, tree: ast.AST) -> List[Tuple[str, Optional[MutationLocation]]]:
        """Replace arithmetic operator - generates ALL possible replacements."""
        # Define ALL possible replacements for each operator
        replacements = {
            ast.Add: [ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod],
            ast.Sub: [ast.Add, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod],
            ast.Mult: [ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mod],
            ast.Div: [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod],
            ast.FloorDiv: [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod],
            ast.Mod: [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv],
        }
        
        # Step 1: Find ALL locations with arithmetic operators
        class LocationFinder(ast.NodeVisitor):
            def __init__(self):
                self.locations = []
            
            def visit_BinOp(self, node):
                op_type = type(node.op)
                if op_type in replacements:
                    self.locations.append((op_type, node.lineno, node.col_offset))
                self.generic_visit(node)
        
        finder = LocationFinder()
        finder.visit(tree)
        
        if not finder.locations:
            return []
        
        # Step 2: Generate ALL mutations for ALL locations
        results = []
        
        for op_type, lineno, col_offset in finder.locations:
            possible_replacements = replacements[op_type]
            
            for new_op_type in possible_replacements:
                class SingleReplacer(ast.NodeTransformer):
                    def __init__(self, target_line, target_col, old_op, new_op):
                        self.target_line = target_line
                        self.target_col = target_col
                        self.old_op = old_op
                        self.new_op = new_op
                        self.replaced = False
                    
                    def visit_BinOp(self, node):
                        if (not self.replaced and 
                            node.lineno == self.target_line and 
                            node.col_offset == self.target_col and
                            isinstance(node.op, self.old_op)):
                            node.op = self.new_op()
                            self.replaced = True
                            return node
                        return self.generic_visit(node)
                
                replacer = SingleReplacer(lineno, col_offset, op_type, new_op_type)
                new_tree = replacer.visit(copy.deepcopy(tree))
                
                if replacer.replaced:
                    mutated_code = self._unparse(new_tree)
                    location = MutationLocation(line=lineno, column=col_offset)
                    results.append((mutated_code, location))
        
        return results
    
    def _replace_logical_op(self, code: str, tree: ast.AST) -> List[Tuple[str, Optional[MutationLocation]]]:
        """Replace logical operator - generates ALL possible replacements."""
        results = []
        
        # Handle BoolOp (and/or)
        class BoolOpFinder(ast.NodeVisitor):
            def __init__(self):
                self.locations = []
            
            def visit_BoolOp(self, node):
                self.locations.append((type(node.op), node.lineno, node.col_offset))
                self.generic_visit(node)
        
        bool_finder = BoolOpFinder()
        bool_finder.visit(tree)
        
        for op_type, lineno, col_offset in bool_finder.locations:
            new_op = ast.Or if op_type == ast.And else ast.And
            
            class BoolReplacer(ast.NodeTransformer):
                def __init__(self, target_line, target_col, old_op, new_op):
                    self.target_line = target_line
                    self.target_col = target_col
                    self.old_op = old_op
                    self.new_op = new_op
                    self.replaced = False
                
                def visit_BoolOp(self, node):
                    if (not self.replaced and 
                        node.lineno == self.target_line and 
                        node.col_offset == self.target_col and
                        isinstance(node.op, self.old_op)):
                        node.op = self.new_op()
                        self.replaced = True
                        return node
                    return self.generic_visit(node)
            
            replacer = BoolReplacer(lineno, col_offset, op_type, new_op)
            new_tree = replacer.visit(copy.deepcopy(tree))
            
            if replacer.replaced:
                mutated_code = self._unparse(new_tree)
                location = MutationLocation(line=lineno, column=col_offset)
                results.append((mutated_code, location))
        
        # Handle Compare operators - generate ALL possible replacements
        compare_replacements = {
            ast.Eq: [ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
            ast.NotEq: [ast.Eq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
            ast.Lt: [ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
            ast.LtE: [ast.Lt, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
            ast.Gt: [ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq],
            ast.GtE: [ast.Gt, ast.Lt, ast.LtE, ast.Eq, ast.NotEq],
        }
        
        class CompareFinder(ast.NodeVisitor):
            def __init__(self):
                self.locations = []
            
            def visit_Compare(self, node):
                if len(node.ops) == 1:
                    op_type = type(node.ops[0])
                    if op_type in compare_replacements:
                        self.locations.append((op_type, node.lineno, node.col_offset))
                self.generic_visit(node)
        
        compare_finder = CompareFinder()
        compare_finder.visit(tree)
        
        for op_type, lineno, col_offset in compare_finder.locations:
            possible_replacements = compare_replacements[op_type]
            
            for new_op_type in possible_replacements:
                class CompareReplacer(ast.NodeTransformer):
                    def __init__(self, target_line, target_col, old_op, new_op):
                        self.target_line = target_line
                        self.target_col = target_col
                        self.old_op = old_op
                        self.new_op = new_op
                        self.replaced = False
                    
                    def visit_Compare(self, node):
                        if (not self.replaced and 
                            node.lineno == self.target_line and 
                            node.col_offset == self.target_col and
                            len(node.ops) == 1 and
                            isinstance(node.ops[0], self.old_op)):
                            node.ops[0] = self.new_op()
                            self.replaced = True
                            return node
                        return self.generic_visit(node)
                
                replacer = CompareReplacer(lineno, col_offset, op_type, new_op_type)
                new_tree = replacer.visit(copy.deepcopy(tree))
                
                if replacer.replaced:
                    mutated_code = self._unparse(new_tree)
                    location = MutationLocation(line=lineno, column=col_offset)
                    results.append((mutated_code, location))
        
        return results
    
    def _remove_function_call(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Remove function call."""
        class Remover(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            def visit_Expr(self, node):
                if not self.done and isinstance(node.value, ast.Call):
                    self.done = True
                    self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                    return ast.Pass()
                return node
        
        remover = Remover()
        new_tree = remover.visit(copy.deepcopy(tree))
        return self._unparse(new_tree), remover.location
    
    def _delete_statement(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """Delete statement."""
        class Deleter(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            def _delete_from_body(self, body):
                if self.done or len(body) < 2:
                    return body
                for i, stmt in enumerate(body):
                    if not isinstance(stmt, (ast.Return, ast.FunctionDef, ast.ClassDef)):
                        self.done = True
                        self.location = MutationLocation(line=stmt.lineno, column=stmt.col_offset)
                        return body[:i] + body[i+1:]
                return body
            def visit_Module(self, node):
                node.body = self._delete_from_body(node.body)
                return self.generic_visit(node)
            def visit_FunctionDef(self, node):
                if not self.done:
                    node.body = self._delete_from_body(node.body)
                return self.generic_visit(node)
        
        deleter = Deleter()
        new_tree = deleter.visit(copy.deepcopy(tree))
        return self._unparse(new_tree), deleter.location
    
    def _replace_relational_op(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        """ROR mutation."""
        all_relational = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
        
        class Replacer(ast.NodeTransformer):
            def __init__(self):
                self.done = False
                self.location = None
            def visit_Compare(self, node):
                if not self.done and len(node.ops) >= 1:
                    for i, op in enumerate(node.ops):
                        op_type = type(op)
                        if op_type in all_relational:
                            replacements = [r for r in all_relational if r != op_type]
                            if replacements:
                                node.ops[i] = random.choice(replacements)()
                                self.done = True
                                self.location = MutationLocation(line=node.lineno, column=node.col_offset)
                                return node
                return self.generic_visit(node)
        
        replacer = Replacer()
        new_tree = replacer.visit(copy.deepcopy(tree))
        return self._unparse(new_tree), replacer.location
    
    def _replace_relational_op_all(self, code: str, tree: ast.AST) -> List[Tuple[str, Optional[MutationLocation]]]:
        """ROR-ALL mutation - generates ALL possible relational operator replacements."""
        all_relational = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
        
        # Find all comparison locations
        class CompareFinder(ast.NodeVisitor):
            def __init__(self):
                self.locations = []
            
            def visit_Compare(self, node):
                if len(node.ops) >= 1:
                    for i, op in enumerate(node.ops):
                        op_type = type(op)
                        if op_type in all_relational:
                            self.locations.append((op_type, i, node.lineno, node.col_offset))
                self.generic_visit(node)
        
        finder = CompareFinder()
        finder.visit(tree)
        
        if not finder.locations:
            return []
        
        results = []
        
        for op_type, op_index, lineno, col_offset in finder.locations:
            # Generate mutations for ALL other relational operators
            possible_replacements = [r for r in all_relational if r != op_type]
            
            for new_op_type in possible_replacements:
                class CompareReplacer(ast.NodeTransformer):
                    def __init__(self, target_line, target_col, target_index, old_op, new_op):
                        self.target_line = target_line
                        self.target_col = target_col
                        self.target_index = target_index
                        self.old_op = old_op
                        self.new_op = new_op
                        self.replaced = False
                    
                    def visit_Compare(self, node):
                        if (not self.replaced and 
                            node.lineno == self.target_line and 
                            node.col_offset == self.target_col and
                            len(node.ops) > self.target_index and
                            isinstance(node.ops[self.target_index], self.old_op)):
                            node.ops[self.target_index] = self.new_op()
                            self.replaced = True
                            return node
                        return self.generic_visit(node)
                
                replacer = CompareReplacer(lineno, col_offset, op_index, op_type, new_op_type)
                new_tree = replacer.visit(copy.deepcopy(tree))
                
                if replacer.replaced:
                    mutated_code = self._unparse(new_tree)
                    location = MutationLocation(line=lineno, column=col_offset)
                    results.append((mutated_code, location))
        
        return results
    
    # Remaining transformers (30-55) - compact implementations
    
    def _replace_power_op(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_BinOp(self, n):
                if not self.done and isinstance(n.op, ast.Pow):
                    n.op = ast.Mult()
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _replace_bitshift_op(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_BinOp(self, n):
                if not self.done:
                    if isinstance(n.op, ast.LShift):
                        n.op = ast.RShift()
                        self.done = True
                        self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                    elif isinstance(n.op, ast.RShift):
                        n.op = ast.LShift()
                        self.done = True
                        self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _replace_bitwise_op(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        replacements = {ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd, ast.BitXor: ast.BitAnd}
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_BinOp(self, n):
                if not self.done:
                    for old, new in replacements.items():
                        if isinstance(n.op, old):
                            n.op = new()
                            self.done = True
                            self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                            return n
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _remove_not(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_UnaryOp(self, n):
                if not self.done and isinstance(n.op, ast.Not):
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                    return self.visit(n.operand)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _negate_unary(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_UnaryOp(self, n):
                if not self.done:
                    if isinstance(n.op, ast.UAdd):
                        n.op = ast.USub()
                        self.done = True
                        self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                    elif isinstance(n.op, ast.USub):
                        n.op = ast.UAdd()
                        self.done = True
                        self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _replace_is_op(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Compare(self, n):
                if not self.done:
                    for i, op in enumerate(n.ops):
                        if isinstance(op, ast.Is):
                            n.ops[i] = ast.IsNot()
                            self.done = True
                            self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                            return n
                        elif isinstance(op, ast.IsNot):
                            n.ops[i] = ast.Is()
                            self.done = True
                            self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                            return n
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _replace_in_op(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Compare(self, n):
                if not self.done:
                    for i, op in enumerate(n.ops):
                        if isinstance(op, ast.In):
                            n.ops[i] = ast.NotIn()
                            self.done = True
                            self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                            return n
                        elif isinstance(op, ast.NotIn):
                            n.ops[i] = ast.In()
                            self.done = True
                            self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                            return n
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _swap_bool_constant(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Constant(self, n):
                if not self.done and isinstance(n.value, bool):
                    n.value = not n.value
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _swap_zero_one(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Constant(self, n):
                if not self.done and n.value in (0, 1) and isinstance(n.value, int) and not isinstance(n.value, bool):
                    n.value = 1 if n.value == 0 else 0
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _mutate_empty_list(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_List(self, n):
                if not self.done and len(n.elts) == 0:
                    n.elts = [ast.Constant(value=1)]
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _mutate_empty_dict(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Dict(self, n):
                if not self.done and len(n.keys) == 0:
                    n.keys = [ast.Constant(value='k')]
                    n.values = [ast.Constant(value=1)]
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _mutate_empty_tuple(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Tuple(self, n):
                if not self.done and len(n.elts) == 0:
                    n.elts = [ast.Constant(value=1)]
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _if_cond_to_false(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_If(self, n):
                if not self.done:
                    n.test = ast.Constant(value=False)
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _remove_if_body(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_If(self, n):
                if not self.done:
                    n.body = [ast.Pass()]
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _remove_else_body(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_If(self, n):
                if not self.done and n.orelse:
                    if not (len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If)):
                        n.orelse = [ast.Pass()]
                        self.done = True
                        self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _while_cond_to_false(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_While(self, n):
                if not self.done:
                    n.test = ast.Constant(value=False)
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _for_to_empty_iter(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_For(self, n):
                if not self.done:
                    n.iter = ast.List(elts=[], ctx=ast.Load())
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _return_to_none(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Return(self, n):
                if not self.done:
                    if not (isinstance(n.value, ast.Constant) and n.value.value is None):
                        n.value = ast.Constant(value=None)
                        self.done = True
                        self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _return_none_to_one(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Return(self, n):
                if not self.done:
                    is_none = n.value is None or (isinstance(n.value, ast.Constant) and n.value.value is None)
                    if is_none:
                        n.value = ast.Constant(value=1)
                        self.done = True
                        self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _remove_raise(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Raise(self, n):
                if not self.done:
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                    return ast.Pass()
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _remove_except_body(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_ExceptHandler(self, n):
                if not self.done:
                    n.body = [ast.Raise()]
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _remove_break(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Break(self, n):
                if not self.done:
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                    return ast.Pass()
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _remove_continue(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Continue(self, n):
                if not self.done:
                    self.done = True
                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                    return ast.Pass()
                return n
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _len_check_to_false(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_If(self, n):
                if not self.done:
                    test = n.test
                    if isinstance(test, ast.Call):
                        if isinstance(test.func, ast.Name) and test.func.id == 'len':
                            n.test = ast.Constant(value=False)
                            self.done = True
                            self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _len_eq_zero_to_true(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Compare(self, n):
                if not self.done and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq):
                    if isinstance(n.left, ast.Call):
                        if isinstance(n.left.func, ast.Name) and n.left.func.id == 'len':
                            if len(n.comparators) == 1:
                                comp = n.comparators[0]
                                if isinstance(comp, ast.Constant) and comp.value == 0:
                                    self.done = True
                                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                                    return ast.Constant(value=True)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location
    
    def _len_gt_zero_to_false(self, code: str, tree: ast.AST) -> Tuple[str, Optional[MutationLocation]]:
        class R(ast.NodeTransformer):
            def __init__(self): self.done, self.location = False, None
            def visit_Compare(self, n):
                if not self.done and len(n.ops) == 1 and isinstance(n.ops[0], ast.Gt):
                    if isinstance(n.left, ast.Call):
                        if isinstance(n.left.func, ast.Name) and n.left.func.id == 'len':
                            if len(n.comparators) == 1:
                                comp = n.comparators[0]
                                if isinstance(comp, ast.Constant) and comp.value == 0:
                                    self.done = True
                                    self.location = MutationLocation(line=n.lineno, column=n.col_offset)
                                    return ast.Constant(value=False)
                return self.generic_visit(n)
        r = R()
        return self._unparse(r.visit(copy.deepcopy(tree))), r.location


# ============================================================================
# MUTATION COMBINER (for higher-order mutations)
# ============================================================================

class MutationCombiner:
    """
    Combines multiple mutations for higher-order mutation testing.
    
    Usage:
        combiner = MutationCombiner(engine)
        results = combiner.generate_combinations(code, mutations, k=2)
    """
    
    def __init__(self, engine: MutationEngine):
        self.engine = engine
    
    def combine_mutations(self, code: str, mutations: List[int], 
                      skip_conflicts: bool = True,
                      line_range: Optional[Tuple[int, int]] = None) -> CombinedMutationResult:
        """
        Apply multiple mutations to code sequentially.
        
        Args:
            code: Source code to mutate
            mutations: List of mutation indices to apply
            skip_conflicts: If True, skip mutations that can't be applied
            
        Returns:
            CombinedMutationResult with the combined mutant
        """
        current_code = code
        applied = []
        applied_names = []
        applied_locations = []  # Track locations for each applied mutation
        skipped = []
        conflicts = []
        
        for mut_idx in mutations:
            if mut_idx < 0 or mut_idx >= len(self.engine.templates):
                skipped.append(mut_idx)
                conflicts.append(f"Invalid mutation index: {mut_idx}")
                continue
            
            template = self.engine.templates[mut_idx]
            
            try:
                tree = ast.parse(current_code)
            except SyntaxError:
                conflicts.append(f"Code became unparseable before applying {template.name}")
                break
            
            if not template.precondition_checker(tree):
                if skip_conflicts:
                    skipped.append(mut_idx)
                    continue
                else:
                    return CombinedMutationResult(
                        original_code=code,
                        mutated_code=current_code,
                        applied_mutations=applied,
                        mutation_names=applied_names,
                        is_valid=False,
                        locations=applied_locations,
                        conflicts=[f"Cannot apply {template.name}: precondition not met"],
                        skipped_mutations=skipped
                    )
            
            try:
                result = template.transformer(current_code, tree)
                
                # Handle both List[Tuple] (multi-mutation) and Tuple (single-mutation)
                location = None
                if isinstance(result, list):
                    # Multi-mutation transformer - pick FIRST mutation
                    if not result:
                        # Empty list - transformer found nothing
                        skipped.append(mut_idx)
                        continue
                    mutated, location = result[0]  # Pick first mutation
                elif isinstance(result, tuple):
                    # Single-mutation transformer - use directly
                    mutated, location = result
                else:
                    # Old-style transformer (just code, no location)
                    mutated = result


                # Check if mutation is within the allowed line range
                if line_range and location:
                    start_line, end_line = line_range
                    if not (start_line <= location.line <= end_line):
                        skipped.append(mut_idx)
                        conflicts.append(f"{template.name} at L{location.line} outside scope ({start_line}-{end_line})")
                        continue



                if mutated.strip() == current_code.strip():
                    skipped.append(mut_idx)
                    continue
                
                try:
                    ast.parse(mutated)
                    current_code = mutated
                    applied.append(mut_idx)
                    applied_names.append(template.name)
                    if location:
                        applied_locations.append(location)
                except SyntaxError as e:
                    skipped.append(mut_idx)
                    conflicts.append(f"{template.name} produced invalid syntax")
                    
            except Exception as e:
                skipped.append(mut_idx)
                conflicts.append(f"{template.name} failed: {e}")
        
        return CombinedMutationResult(
            original_code=code,
            mutated_code=current_code,
            applied_mutations=applied,
            mutation_names=applied_names,
            is_valid=len(applied) > 0 and current_code != code,
            locations=applied_locations,
            conflicts=conflicts,
            skipped_mutations=skipped
        )
    
    def generate_combinations(self, code: str, 
                           applicable_mutations: List[int],
                           k: int,
                           max_combinations: int = 100,
                           line_range: Optional[Tuple[int, int]] = None) -> List[CombinedMutationResult]:
        """
        Generate all valid k-mutation combinations.
        
        Args:
            code: Source code to mutate
            mutations: List of mutation indices to use
            k: Number of mutations to combine
            max_combinations: Maximum combinations to generate
            
        Returns:
            List of CombinedMutationResult objects
        """
        results = []
        combinations = list(itertools.combinations(applicable_mutations, k))
        
        if len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
        
        for combo in combinations:
            for perm in itertools.permutations(combo):
                result = self.combine_mutations(code, list(perm), skip_conflicts=True, line_range=line_range)
                
                if len(result.applied_mutations) == k and result.is_valid:
                    results.append(result)
                    break
        
        return results
    
    def generate_up_to_k(self, code: str,
                                       applicable_mutations: List[int],
                                       max_k: int,
                                       max_per_k: int = 50,
                                       line_range: Optional[Tuple[int, int]] = None) -> Dict[int, List[CombinedMutationResult]]:
        """
        Generate combinations for k=1, 2, 3, ..., max_k.
        
        Args:
            code: Source code to mutate
            mutations: Mutations to combine
            max_k: Maximum number of mutations to combine
            max_per_k: Maximum results per k value
            
        Returns:
            Dict mapping k -> list of CombinedMutationResult
        """
        results = {}
        
        for k in range(1, max_k + 1):
            k_results = self.generate_combinations(code, applicable_mutations, k, max_combinations=max_per_k,line_range=line_range)
            results[k] = k_results
            
            if not k_results:
                break
        
        return results
    
    def generate_combinations_from_mutants(self,
                                          code: str,
                                          k: int,
                                          mutations_to_use: Optional[List[int]] = None,
                                          max_combinations: int = 1000,
                                          line_range: Optional[Tuple[int, int]] = None) -> List[CombinedMutationResult]:
        """
        Generate ALL possible combinations from individual mutants (not operators).
        
        This is the key difference from generate_combinations():
        - generate_combinations: Combines mutation OPERATORS (picks first from each)
        - generate_combinations_from_mutants: Combines actual MUTANTS (all combinations)
        
        Example:
            arithmetic_op has 5 mutations: [m1, m2, m3, m4, m5]
            if_true has 1 mutation: [m6]
            
            Old way: Combines operator indices [23, 42] → 1 combination (m1 + m6)
            New way: Combines all mutants → 5 combinations (m1+m6, m2+m6, m3+m6, m4+m6, m5+m6)
        
        Args:
            code: Source code to mutate
            k: Number of mutations to combine
            mutations_to_use: Optional list of mutation operator indices to use
            max_combinations: Maximum combinations to generate
            line_range: Optional line range filter
            
        Returns:
            List of ALL valid k-combinations of actual mutants
        """
        # Step 1: Generate ALL k=1 mutations first
        if mutations_to_use:
            first_order = self.engine.mutate(code, mutations=mutations_to_use, line_range=line_range)
        else:
            first_order = self.engine.mutate(code, line_range=line_range)
        
        if k == 1:
            # Convert to CombinedMutationResult for consistency
            return [CombinedMutationResult(
                original_code=m.original_code,
                mutated_code=m.mutated_code,
                applied_mutations=[self.engine.get_mutation_index(m.pattern_name)] if m.pattern_name else [],
                mutation_names=[m.pattern_name],
                is_valid=m.is_valid,
                locations=[m.location] if m.location else [],
                conflicts=[],
                skipped_mutations=[]
            ) for m in first_order if m.is_valid]
        
        # Step 2: For k>1, combine SPECIFIC mutants by applying their patterns
        results = []
        attempted = 0
        
        # Generate all k-combinations of the first-order mutations
        for combo in itertools.combinations(first_order, k):
            if attempted >= max_combinations:
                break
            
            attempted += 1
            
            # Try to apply all mutations in this combination
            result = self._apply_mutation_combination(code, combo, line_range)
            
            if result and result.is_valid and len(result.applied_mutations) == k:
                results.append(result)
        
        return results
    
    def _apply_mutation_combination(self,
                                   code: str,
                                   mutations: Tuple['MutationResult', ...],
                                   line_range: Optional[Tuple[int, int]] = None) -> Optional[CombinedMutationResult]:
        """
        Apply a combination of specific mutations sequentially.
        
        Args:
            code: Original source code
            mutations: Tuple of MutationResult objects to combine
            line_range: Optional line range filter
            
        Returns:
            CombinedMutationResult if successful, None otherwise
        """
        current_code = code
        applied = []
        applied_names = []
        applied_locations = []
        skipped = []
        conflicts = []
        
        for mutation in mutations:
            # Get the mutation operator index
            mut_idx = self.engine.get_mutation_index(mutation.pattern_name)
            if mut_idx is None:
                skipped.append(-1)
                conflicts.append(f"Unknown pattern: {mutation.pattern_name}")
                continue
            
            template = self.engine.templates[mut_idx]
            
            # Try to parse current code
            try:
                tree = ast.parse(current_code)
            except SyntaxError:
                conflicts.append(f"Code became unparseable before {mutation.pattern_name}")
                break
            
            # Check if pattern is still applicable
            if not template.precondition_checker(tree):
                skipped.append(mut_idx)
                continue
            
            # Apply the mutation
            try:
                result = template.transformer(current_code, tree)
                
                # Handle both list and tuple returns
                location = None
                if isinstance(result, list):
                    if not result:
                        skipped.append(mut_idx)
                        continue
                    # Find the mutation closest to the original location
                    mutated, location = self._select_matching_mutation(result, mutation)
                elif isinstance(result, tuple):
                    mutated, location = result
                else:
                    mutated = result
                
                # Check line range
                if line_range and location:
                    start_line, end_line = line_range
                    if not (start_line <= location.line <= end_line):
                        skipped.append(mut_idx)
                        conflicts.append(f"{mutation.pattern_name} at L{location.line} outside scope")
                        continue
                
                # Validate change
                if mutated.strip() == current_code.strip():
                    skipped.append(mut_idx)
                    continue
                
                # Validate syntax
                try:
                    ast.parse(mutated)
                    current_code = mutated
                    applied.append(mut_idx)
                    applied_names.append(mutation.pattern_name)
                    if location:
                        applied_locations.append(location)
                except SyntaxError:
                    skipped.append(mut_idx)
                    conflicts.append(f"{mutation.pattern_name} produced invalid syntax")
                    
            except Exception as e:
                skipped.append(mut_idx)
                conflicts.append(f"{mutation.pattern_name} failed: {e}")
        
        return CombinedMutationResult(
            original_code=code,
            mutated_code=current_code,
            applied_mutations=applied,
            mutation_names=applied_names,
            is_valid=len(applied) > 0 and current_code != code,
            locations=applied_locations,
            conflicts=conflicts,
            skipped_mutations=skipped
        )
    
    def _select_matching_mutation(self,
                                 mutations: List[Tuple[str, Optional[MutationLocation]]],
                                 original: 'MutationResult') -> Tuple[str, Optional[MutationLocation]]:
        """
        Select the mutation from the list that best matches the original.
        
        For now, selects the first mutation. Could be improved to match by location.
        """
        if not mutations:
            return "", None
        
        # Simple strategy: pick first
        # Better strategy: match by line/column if original.location is available
        if original.location and len(mutations) > 1:
            # Try to find mutation at same location
            for mutated, loc in mutations:
                if loc and loc.line == original.location.line:
                    return mutated, loc
        
        return mutations[0]