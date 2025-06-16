use core::ops::Deref;
use std::collections::{BTreeMap, HashMap, HashSet};

use itertools::izip;
use lang_c::ast;

use crate::ir::*;
use crate::opt::opt_utils::*;
use crate::opt::*;

pub type Gvn = FunctionPass<GvnInner>;

#[derive(Default, Clone, Copy, Debug)]
pub struct GvnInner {}

// Represents an expression that can be value-numbered
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ValueExpr {
    BinOp {
        op: ast::BinaryOperator,
        lhs: Operand,
        rhs: Operand,
    },
    UnaryOp {
        op: ast::UnaryOperator,
        operand: Operand,
    },
    TypeCast {
        value: Operand,
        target_dtype: Dtype,
    },
    GetElementPtr {
        ptr: Operand,
        offset: Operand,
        dtype: Dtype,
    },
}

impl ValueExpr {
    // Normalize commutative operations by ordering operands
    fn normalize(self) -> Self {
        match self {
            ValueExpr::BinOp { op, lhs, rhs } => {
                match op {
                    ast::BinaryOperator::Plus
                    | ast::BinaryOperator::Multiply
                    | ast::BinaryOperator::BitwiseAnd
                    | ast::BinaryOperator::BitwiseXor
                    | ast::BinaryOperator::BitwiseOr
                    | ast::BinaryOperator::Equals
                    | ast::BinaryOperator::NotEquals => {
                        // For commutative operations, order operands deterministically
                        if format!("{:?}", lhs) > format!("{:?}", rhs) {
                            ValueExpr::BinOp {
                                op,
                                lhs: rhs,
                                rhs: lhs,
                            }
                        } else {
                            ValueExpr::BinOp { op, lhs, rhs }
                        }
                    }
                    _ => ValueExpr::BinOp { op, lhs, rhs },
                }
            }
            other => other,
        }
    }
}

impl Optimize<FunctionDefinition> for GvnInner {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let cfg = make_cfg(code);
        let reverse_cfg = reverse_cfg(&cfg);
        let domtree = Domtree::new(code.bid_init, &cfg, &reverse_cfg);

        let mut changed = false;

        // Maps registers to their replacement operands
        let mut replacements: HashMap<RegisterId, Operand> = HashMap::new();

        // Maps expressions to their first occurrence (register)
        let mut expr_to_first_occurrence: HashMap<ValueExpr, RegisterId> = HashMap::new();

        // Process blocks in reverse post-order to ensure we process dominators before dominated blocks
        let post_order = traverse_postorder(code.bid_init, &cfg);
        let mut rpo = post_order;
        rpo.reverse();

        // Phase 1: Analyze constant propagation through block parameters
        // Run this phase to fixed point with a limit to prevent infinite loops
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 30;

        loop {
            iterations += 1;
            let mut changed_this_round = false;

            changed_this_round |= propagate_constants_through_phi_nodes(
                &rpo,
                &code.blocks,
                &reverse_cfg,
                &mut replacements,
            );

            // Phase 2: GVN
            changed_this_round |= find_redundant_expressions(
                &rpo,
                &code.blocks,
                &domtree,
                &mut replacements,
                &mut expr_to_first_occurrence,
            );

            // Phase 3
            let phi_nodes_to_add = generate_phi_nodes_for_expressions(
                &rpo,
                &code.blocks,
                &reverse_cfg,
                &domtree,
                &mut replacements,
                &mut changed_this_round,
            );

            // Phase 4
            if !phi_nodes_to_add.is_empty() {
                add_phi_nodes(code, phi_nodes_to_add);
                changed_this_round = true;
            }

            // Phase 5
            changed_this_round |=
                propagate_phi_equivalences(&rpo, &code.blocks, &reverse_cfg, &mut replacements);

            if !changed_this_round || iterations >= MAX_ITERATIONS {
                break;
            }

            // Phase 6: Apply replacements if any were found
            if !replacements.is_empty() {
                apply_replacements(code, &replacements);
            }

            changed = true;
        }

        changed
    }
}

fn propagate_constants_through_phi_nodes(
    rpo: &[BlockId],
    blocks: &BTreeMap<BlockId, Block>,
    reverse_cfg: &HashMap<BlockId, Vec<(BlockId, JumpArg)>>,
    replacements: &mut HashMap<RegisterId, Operand>,
) -> bool {
    let mut changed = false;

    for &bid in rpo {
        if let Some(block) = blocks.get(&bid) {
            // For each phi node, check if all predecessors pass the same constant
            for (phi_idx, _phi) in block.phinodes.iter().enumerate() {
                if let Some(preds) = reverse_cfg.get(&bid) {
                    let mut constant_value: Option<Operand> = None;
                    let mut all_same = true;

                    for (pred_bid, _) in preds {
                        if let Some(pred_block) = blocks.get(pred_bid) {
                            match &pred_block.exit {
                                BlockExit::Jump { arg } => {
                                    if arg.bid == bid && phi_idx < arg.args.len() {
                                        // Apply replacements to see the actual value being passed
                                        let arg_val =
                                            apply_replacement(&arg.args[phi_idx], replacements);
                                        if let Some(ref const_val) = constant_value {
                                            if &arg_val != const_val {
                                                all_same = false;
                                                break;
                                            }
                                        } else {
                                            constant_value = Some(arg_val);
                                        }
                                    }
                                }
                                BlockExit::ConditionalJump {
                                    arg_then, arg_else, ..
                                } => {
                                    let arg = if arg_then.bid == bid {
                                        arg_then
                                    } else if arg_else.bid == bid {
                                        arg_else
                                    } else {
                                        continue;
                                    };

                                    if phi_idx < arg.args.len() {
                                        // Apply replacements to see the actual value being passed
                                        let arg_val =
                                            apply_replacement(&arg.args[phi_idx], replacements);
                                        if let Some(ref const_val) = constant_value {
                                            if &arg_val != const_val {
                                                all_same = false;
                                                break;
                                            }
                                        } else {
                                            constant_value = Some(arg_val);
                                        }
                                    }
                                }
                                BlockExit::Switch { default, cases, .. } => {
                                    // Check default case
                                    if default.bid == bid && phi_idx < default.args.len() {
                                        // Apply replacements to see the actual value being passed
                                        let arg_val =
                                            apply_replacement(&default.args[phi_idx], replacements);
                                        if let Some(ref const_val) = constant_value {
                                            if &arg_val != const_val {
                                                all_same = false;
                                                break;
                                            }
                                        } else {
                                            constant_value = Some(arg_val);
                                        }
                                    }

                                    // Check all switch cases
                                    for (_, case_arg) in cases {
                                        if case_arg.bid == bid && phi_idx < case_arg.args.len() {
                                            // Apply replacements to see the actual value being passed
                                            let arg_val = apply_replacement(
                                                &case_arg.args[phi_idx],
                                                replacements,
                                            );
                                            if let Some(ref const_val) = constant_value {
                                                if &arg_val != const_val {
                                                    all_same = false;
                                                    break;
                                                }
                                            } else {
                                                constant_value = Some(arg_val);
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }

                    // If all predecessors pass the same value, replace the phi node with that value
                    if all_same && constant_value.is_some() {
                        let rid = RegisterId::arg(bid, phi_idx);
                        let value = constant_value.unwrap();
                        let _unused = replacements.insert(rid, value);
                        changed = true;
                    }
                }
            }
        }
    }

    changed
}

fn apply_replacement(operand: &Operand, replacements: &HashMap<RegisterId, Operand>) -> Operand {
    apply_replacement_with_visited(operand, replacements, &mut HashSet::new())
}

fn apply_replacement_with_visited(
    operand: &Operand,
    replacements: &HashMap<RegisterId, Operand>,
    visited: &mut HashSet<RegisterId>,
) -> Operand {
    if let Some((rid, _)) = operand.get_register() {
        if let Some(replacement) = replacements.get(rid) {
            // Check for cycles
            if !visited.insert(*rid) {
                // We've already visited this register, return the original to avoid infinite recursion
                return operand.clone();
            }
            // Recursively apply replacements
            let result = apply_replacement_with_visited(replacement, replacements, visited);
            let _unused = visited.remove(rid);
            return result;
        }
    }
    operand.clone()
}

fn dominates_register(domtree: &Domtree, def_rid: RegisterId, use_bid: BlockId) -> bool {
    match def_rid {
        RegisterId::Temp { bid, iid } => {
            if bid == use_bid {
                // In the same block, definition must come before use
                // This is handled by the caller ensuring we don't use forward references
                true
            } else {
                // Check if bid dominates use_bid
                let mut current = use_bid;
                while let Some(idom) = domtree.idom(current) {
                    if idom == bid {
                        return true;
                    }
                    current = idom;
                }
                false
            }
        }
        _ => true, // Arguments always available
    }
}

fn find_value_in_block(
    expr: &ValueExpr,
    bid: BlockId,
    blocks: &BTreeMap<BlockId, Block>,
    replacements: &HashMap<RegisterId, Operand>,
) -> Option<Operand> {
    // First, check if the value is computed in this block
    let block = blocks.get(&bid)?;

    for (i, instr) in block.instructions.iter().enumerate() {
        let rid = RegisterId::temp(bid, i);

        // Check if this register was replaced
        if let Some(replacement) = replacements.get(&rid) {
            // Check if the replacement computes our expression
            if let Some((rep_rid, dtype)) = replacement.get_register() {
                if let RegisterId::Temp {
                    bid: rep_bid,
                    iid: rep_iid,
                } = rep_rid
                {
                    if let Some(rep_block) = blocks.get(rep_bid) {
                        if let Some(rep_instr) = rep_block.instructions.get(*rep_iid) {
                            if matches_expr(rep_instr.deref(), expr, replacements) {
                                return Some(replacement.clone());
                            }
                        }
                    }
                }
            }
        }

        // Check if this instruction computes our expression
        if matches_expr(instr.deref(), expr, replacements) {
            return Some(Operand::register(rid, instr.dtype()));
        }
    }

    None
}

fn find_value_in_block_or_dominators(
    expr: &ValueExpr,
    bid: BlockId,
    blocks: &BTreeMap<BlockId, Block>,
    domtree: &Domtree,
    replacements: &HashMap<RegisterId, Operand>,
) -> Option<Operand> {
    // First check the block itself
    if let Some(value) = find_value_in_block(expr, bid, blocks, replacements) {
        return Some(value);
    }

    // Then check dominating blocks
    let mut current = bid;
    while let Some(idom) = domtree.idom(current) {
        if let Some(value) = find_value_in_block(expr, idom, blocks, replacements) {
            // Make sure the value dominates the original block
            if let Some((rid, _)) = value.get_register() {
                if dominates_register(domtree, *rid, bid) {
                    return Some(value);
                }
            } else {
                // Constants always dominate
                return Some(value);
            }
        }
        current = idom;
    }

    None
}

fn matches_expr(
    instr: &Instruction,
    expr: &ValueExpr,
    replacements: &HashMap<RegisterId, Operand>,
) -> bool {
    // Convert instruction to ValueExpr and compare
    let instr_expr = match instr {
        Instruction::BinOp { op, lhs, rhs, .. } => Some(
            ValueExpr::BinOp {
                op: op.clone(),
                lhs: apply_replacement(lhs, replacements),
                rhs: apply_replacement(rhs, replacements),
            }
            .normalize(),
        ),
        Instruction::UnaryOp { op, operand, .. } => Some(ValueExpr::UnaryOp {
            op: op.clone(),
            operand: apply_replacement(operand, replacements),
        }),
        Instruction::TypeCast {
            value,
            target_dtype,
        } => Some(ValueExpr::TypeCast {
            value: apply_replacement(value, replacements),
            target_dtype: target_dtype.clone(),
        }),
        Instruction::GetElementPtr { ptr, offset, dtype } => Some(ValueExpr::GetElementPtr {
            ptr: apply_replacement(ptr, replacements),
            offset: apply_replacement(offset, replacements),
            dtype: dtype.clone(),
        }),
        Instruction::Load { .. } => {
            // Skip load instructions for GVN optimization
            // Load instructions are not safe to optimize without proper alias analysis
            None
        }
        _ => None,
    };

    if let Some(instr_expr_val) = instr_expr {
        &instr_expr_val == expr
    } else {
        false
    }
}

fn find_redundant_expressions(
    rpo: &[BlockId],
    blocks: &BTreeMap<BlockId, Block>,
    domtree: &Domtree,
    replacements: &mut HashMap<RegisterId, Operand>,
    expr_to_first_occurrence: &mut HashMap<ValueExpr, RegisterId>,
) -> bool {
    let mut changed = false;

    for &bid in rpo {
        let block = blocks.get(&bid).unwrap();

        for (i, instr) in block.instructions.iter().enumerate() {
            let rid = RegisterId::temp(bid, i);

            // Convert instruction to ValueExpr if possible
            let expr_opt = instruction_to_expr(instr.deref(), replacements);

            if let Some(expr) = expr_opt {
                // Look for an existing computation we can reuse
                let replacement = find_existing_computation(
                    &expr,
                    bid,
                    rid,
                    blocks,
                    domtree,
                    replacements,
                    instr.dtype(),
                );

                match replacement {
                    Some(operand) => {
                        let _unused = replacements.insert(rid, operand);
                        changed = true;
                    }
                    None => {
                        // No replacement found, record this as the first occurrence
                        let _unused = expr_to_first_occurrence.insert(expr, rid);
                    }
                }
            }
        }
    }

    changed
}

fn instruction_to_expr(
    instr: &Instruction,
    replacements: &HashMap<RegisterId, Operand>,
) -> Option<ValueExpr> {
    match instr {
        Instruction::BinOp { op, lhs, rhs, .. } => {
            let normalized_lhs = apply_replacement(lhs, replacements);
            let normalized_rhs = apply_replacement(rhs, replacements);

            Some(
                ValueExpr::BinOp {
                    op: op.clone(),
                    lhs: normalized_lhs,
                    rhs: normalized_rhs,
                }
                .normalize(),
            )
        }
        Instruction::UnaryOp { op, operand, .. } => {
            let normalized_operand = apply_replacement(operand, replacements);

            Some(ValueExpr::UnaryOp {
                op: op.clone(),
                operand: normalized_operand,
            })
        }
        Instruction::TypeCast {
            value,
            target_dtype,
        } => {
            let normalized_value = apply_replacement(value, replacements);

            Some(ValueExpr::TypeCast {
                value: normalized_value,
                target_dtype: target_dtype.clone(),
            })
        }
        Instruction::GetElementPtr { ptr, offset, dtype } => {
            let normalized_ptr = apply_replacement(ptr, replacements);
            let normalized_offset = apply_replacement(offset, replacements);

            Some(ValueExpr::GetElementPtr {
                ptr: normalized_ptr,
                offset: normalized_offset,
                dtype: dtype.clone(),
            })
        }
        Instruction::Load { .. } => {
            // Skip load instructions for GVN optimization
            // Load instructions are not safe to optimize without proper alias analysis
            None
        }
        _ => None,
    }
}

fn find_existing_computation(
    expr: &ValueExpr,
    current_bid: BlockId,
    current_rid: RegisterId,
    blocks: &BTreeMap<BlockId, Block>,
    domtree: &Domtree,
    replacements: &HashMap<RegisterId, Operand>,
    dtype: Dtype,
) -> Option<Operand> {
    // First check if this expression uses any phi node parameters from the current block
    // If it does, we cannot replace it with a value from a dominating block
    let uses_current_phi = match expr {
        ValueExpr::BinOp { lhs, rhs, .. } => {
            matches_phi_from_block(lhs, current_bid) || matches_phi_from_block(rhs, current_bid)
        }
        ValueExpr::UnaryOp { operand, .. } => matches_phi_from_block(operand, current_bid),
        ValueExpr::TypeCast { value, .. } => matches_phi_from_block(value, current_bid),
        ValueExpr::GetElementPtr { ptr, offset, .. } => {
            matches_phi_from_block(ptr, current_bid) || matches_phi_from_block(offset, current_bid)
        }
    };

    if uses_current_phi {
        return None;
    }

    // Search through all blocks that dominate this one
    let mut search_blocks = vec![current_bid];
    let mut current = current_bid;
    while let Some(idom) = domtree.idom(current) {
        search_blocks.push(idom);
        current = idom;
    }

    // Search in reverse order (from root to current) to find the first occurrence
    search_blocks.reverse();

    for &search_bid in &search_blocks {
        if let Some(search_block) = blocks.get(&search_bid) {
            for (j, search_instr) in search_block.instructions.iter().enumerate() {
                let search_rid = RegisterId::temp(search_bid, j);

                // Don't compare with ourselves
                if search_rid == current_rid {
                    continue;
                }

                // If we're in the same block, stop when we reach the current instruction
                // to avoid forward references
                if search_bid == current_bid {
                    if let RegisterId::Temp {
                        bid: _,
                        iid: current_iid,
                    } = current_rid
                    {
                        if j >= current_iid {
                            break;
                        }
                    }
                }

                // Check if this instruction computes the same expression
                if matches_expr(search_instr.deref(), expr, replacements) {
                    return Some(Operand::register(search_rid, dtype));
                }
            }
        }
    }

    None
}

fn matches_phi_from_block(operand: &Operand, bid: BlockId) -> bool {
    if let Some((rid, _)) = operand.get_register() {
        match rid {
            RegisterId::Arg { bid: arg_bid, .. } => *arg_bid == bid,
            _ => false,
        }
    } else {
        false
    }
}

fn generate_phi_nodes_for_expressions(
    rpo: &[BlockId],
    blocks: &BTreeMap<BlockId, Block>,
    reverse_cfg: &HashMap<BlockId, Vec<(BlockId, JumpArg)>>,
    domtree: &Domtree,
    replacements: &mut HashMap<RegisterId, Operand>,
    changed: &mut bool,
) -> HashMap<BlockId, Vec<(Dtype, Vec<(BlockId, Operand)>)>> {
    let mut phi_nodes_to_add: HashMap<BlockId, Vec<(Dtype, Vec<(BlockId, Operand)>)>> =
        HashMap::new();

    // Check each block for values that could come from multiple predecessors
    for &bid in rpo {
        if let Some(preds) = reverse_cfg.get(&bid) {
            if preds.len() > 1 {
                // This block has multiple predecessors, check for common values
                let block = blocks.get(&bid).unwrap();

                // For each instruction in this block, check if it computes something
                // that's also computed in the predecessors
                for (i, instr) in block.instructions.iter().enumerate() {
                    let rid = RegisterId::temp(bid, i);

                    // Skip if this is already being replaced
                    if replacements.contains_key(&rid) {
                        continue;
                    }

                    // Convert instruction to ValueExpr if possible
                    let expr_opt = instruction_to_expr(instr.deref(), replacements);

                    if let Some(expr) = expr_opt {
                        // Check if this expression is available from predecessors
                        let mut pred_values: Vec<(BlockId, Operand)> = Vec::new();
                        let mut all_same_value = true;
                        let mut first_value: Option<Operand> = None;

                        for (pred_bid, _) in preds {
                            if let Some(value) = find_value_in_block_or_dominators(
                                &expr,
                                *pred_bid,
                                blocks,
                                domtree,
                                replacements,
                            ) {
                                pred_values.push((*pred_bid, value.clone()));

                                // Check if all values are the same
                                if let Some(ref fv) = first_value {
                                    if &value != fv {
                                        all_same_value = false;
                                    }
                                } else {
                                    first_value = Some(value);
                                }
                            }
                        }

                        // If available from all predecessors
                        if pred_values.len() == preds.len() {
                            // If all predecessors provide the exact same value, just use that value
                            if all_same_value && first_value.is_some() {
                                let value = first_value.unwrap();
                                // Only replace if the value is different from the current register
                                if let Some((value_rid, _)) = value.get_register() {
                                    if value_rid != &rid {
                                        let _unused = replacements.insert(rid, value);
                                        *changed = true;
                                    }
                                } else {
                                    // Constants and other operands
                                    let _unused = replacements.insert(rid, value);
                                    *changed = true;
                                }
                            } else {
                                // Different values from predecessors, create a phi node
                                // Calculate the absolute phi index by adding existing phi nodes count
                                let existing_phi_count = block.phinodes.len();
                                let relative_phi_idx =
                                    phi_nodes_to_add.entry(bid).or_default().len();
                                let absolute_phi_idx = existing_phi_count + relative_phi_idx;

                                phi_nodes_to_add
                                    .entry(bid)
                                    .or_default()
                                    .push((instr.dtype(), pred_values));

                                // Replace uses of this instruction with phi node reference
                                let _unused = replacements.insert(
                                    rid,
                                    Operand::register(
                                        RegisterId::arg(bid, absolute_phi_idx),
                                        instr.dtype(),
                                    ),
                                );
                                *changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    phi_nodes_to_add
}

fn apply_replacements(code: &mut FunctionDefinition, replacements: &HashMap<RegisterId, Operand>) {
    for (_, block) in code.blocks.iter_mut() {
        for instr in block.instructions.iter_mut() {
            let _unused = instr.walk(|operand| {
                if let Some((rid, _)) = operand.get_register() {
                    if let Some(replacement) = replacements.get(rid) {
                        *operand = replacement.clone();
                        return true;
                    }
                }
                false
            });
        }

        let _unused = block.exit.walk(|operand| {
            if let Some((rid, _)) = operand.get_register() {
                if let Some(replacement) = replacements.get(rid) {
                    *operand = replacement.clone();
                    return true;
                }
            }
            false
        });
    }
}

fn add_phi_nodes(
    code: &mut FunctionDefinition,
    phi_nodes_to_add: HashMap<BlockId, Vec<(Dtype, Vec<(BlockId, Operand)>)>>,
) {
    // Store the original phi node counts before adding new ones
    let original_phi_counts: HashMap<BlockId, usize> = phi_nodes_to_add
        .keys()
        .map(|bid| (*bid, code.blocks.get(bid).unwrap().phinodes.len()))
        .collect();

    for (bid, phi_list) in &phi_nodes_to_add {
        if let Some(block) = code.blocks.get_mut(&bid) {
            for (dtype, _) in phi_list {
                block.phinodes.push(Named::new(None, dtype.clone()));
            }
        }

        // Update jump arguments in predecessor blocks
        let existing_phi_count = *original_phi_counts.get(&bid).unwrap();
        for (relative_phi_idx, (_, pred_values)) in phi_list.iter().enumerate() {
            let absolute_phi_idx = existing_phi_count + relative_phi_idx;
            for (pred_bid, value) in pred_values {
                if let Some(pred_block) = code.blocks.get_mut(pred_bid) {
                    update_jump_arguments(
                        &mut pred_block.exit,
                        *bid,
                        absolute_phi_idx,
                        value.clone(),
                    );
                }
            }
        }
    }
}

fn update_jump_arguments(
    exit: &mut BlockExit,
    target_bid: BlockId,
    phi_idx: usize,
    value: Operand,
) {
    match exit {
        BlockExit::Jump { arg } => {
            if arg.bid == target_bid {
                while arg.args.len() <= phi_idx {
                    arg.args.push(Operand::constant(Constant::unit()));
                }
                arg.args[phi_idx] = value;
            }
        }
        BlockExit::ConditionalJump {
            arg_then, arg_else, ..
        } => {
            if arg_then.bid == target_bid {
                while arg_then.args.len() <= phi_idx {
                    arg_then.args.push(Operand::constant(Constant::unit()));
                }
                arg_then.args[phi_idx] = value.clone();
            }
            if arg_else.bid == target_bid {
                while arg_else.args.len() <= phi_idx {
                    arg_else.args.push(Operand::constant(Constant::unit()));
                }
                arg_else.args[phi_idx] = value;
            }
        }
        BlockExit::Switch { default, cases, .. } => {
            // Update default case
            if default.bid == target_bid {
                while default.args.len() <= phi_idx {
                    default.args.push(Operand::constant(Constant::unit()));
                }
                default.args[phi_idx] = value.clone();
            }

            // Update all switch cases
            for (_, case_arg) in cases {
                if case_arg.bid == target_bid {
                    while case_arg.args.len() <= phi_idx {
                        case_arg.args.push(Operand::constant(Constant::unit()));
                    }
                    case_arg.args[phi_idx] = value.clone();
                }
            }
        }
        _ => {}
    }
}

// New function to find and propagate equivalences between computed values and phi parameters
fn propagate_phi_equivalences(
    rpo: &[BlockId],
    blocks: &BTreeMap<BlockId, Block>,
    reverse_cfg: &HashMap<BlockId, Vec<(BlockId, JumpArg)>>,
    replacements: &mut HashMap<RegisterId, Operand>,
) -> bool {
    let mut changed = false;

    // For each block
    for &bid in rpo {
        let block = blocks.get(&bid).unwrap();

        // For each phi node in the block
        for (phi_idx, phi) in block.phinodes.iter().enumerate() {
            let phi_rid = RegisterId::arg(bid, phi_idx);

            // Skip if the phi itself was replaced
            if replacements.contains_key(&phi_rid) {
                continue;
            }

            // Check what expression this phi represents by examining what all predecessors pass
            if let Some(preds) = reverse_cfg.get(&bid) {
                // First, determine what expression all predecessors are passing
                let mut common_expr: Option<ValueExpr> = None;
                let mut all_same = true;

                for (pred_bid, _) in preds {
                    if let Some(pred_block) = blocks.get(pred_bid) {
                        // Find what value this predecessor passes to this phi
                        let passed_value = match &pred_block.exit {
                            BlockExit::Jump { arg } => {
                                if arg.bid == bid && phi_idx < arg.args.len() {
                                    Some(&arg.args[phi_idx])
                                } else {
                                    None
                                }
                            }
                            BlockExit::ConditionalJump {
                                arg_then, arg_else, ..
                            } => {
                                if arg_then.bid == bid && phi_idx < arg_then.args.len() {
                                    Some(&arg_then.args[phi_idx])
                                } else if arg_else.bid == bid && phi_idx < arg_else.args.len() {
                                    Some(&arg_else.args[phi_idx])
                                } else {
                                    None
                                }
                            }
                            BlockExit::Switch { default, cases, .. } => {
                                if default.bid == bid && phi_idx < default.args.len() {
                                    Some(&default.args[phi_idx])
                                } else {
                                    cases.iter().find_map(|(_, case_arg)| {
                                        if case_arg.bid == bid && phi_idx < case_arg.args.len() {
                                            Some(&case_arg.args[phi_idx])
                                        } else {
                                            None
                                        }
                                    })
                                }
                            }
                            _ => None,
                        };

                        if let Some(passed_operand) = passed_value {
                            // Apply replacements to see the actual value
                            let actual_value = apply_replacement(passed_operand, replacements);

                            // Check what expression this value represents
                            if let Some((value_rid, _)) = actual_value.get_register() {
                                if let RegisterId::Temp {
                                    bid: value_bid,
                                    iid: value_iid,
                                } = value_rid
                                {
                                    if let Some(value_block) = blocks.get(value_bid) {
                                        if let Some(value_instr) =
                                            value_block.instructions.get(*value_iid)
                                        {
                                            if let Some(expr) = instruction_to_expr(
                                                value_instr.deref(),
                                                replacements,
                                            ) {
                                                if let Some(ref ce) = common_expr {
                                                    if ce != &expr {
                                                        all_same = false;
                                                        break;
                                                    }
                                                } else {
                                                    common_expr = Some(expr);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // If all predecessors pass values that compute the same expression
                if all_same && common_expr.is_some() {
                    let expr = common_expr.unwrap();

                    // Find instructions in this block that compute the same expression
                    for (i, instr) in block.instructions.iter().enumerate() {
                        let rid = RegisterId::temp(bid, i);

                        // Skip if already replaced
                        if replacements.contains_key(&rid) {
                            continue;
                        }

                        // Check type compatibility
                        if instr.dtype() != *phi.deref() {
                            continue;
                        }

                        // Check if this instruction computes the same expression
                        if matches_expr(instr.deref(), &expr, replacements) {
                            // Replace this instruction with the phi
                            let _unused = replacements
                                .insert(rid, Operand::register(phi_rid, phi.deref().clone()));
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    changed
}
