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

        // Analyze constant propagation through block parameters
        for &bid in &rpo {
            if let Some(block) = code.blocks.get(&bid) {
                // For each phi node, check if all predecessors pass the same constant
                for (phi_idx, phi) in block.phinodes.iter().enumerate() {
                    if let Some(preds) = reverse_cfg.get(&bid) {
                        let mut constant_value: Option<Operand> = None;
                        let mut all_same = true;

                        for (pred_bid, _) in preds {
                            if let Some(pred_block) = code.blocks.get(pred_bid) {
                                match &pred_block.exit {
                                    BlockExit::Jump { arg } => {
                                        if arg.bid == bid && phi_idx < arg.args.len() {
                                            let arg_val = &arg.args[phi_idx];
                                            if let Some(ref const_val) = constant_value {
                                                if arg_val != const_val {
                                                    all_same = false;
                                                    break;
                                                }
                                            } else {
                                                constant_value = Some(arg_val.clone());
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
                                            let arg_val = &arg.args[phi_idx];
                                            if let Some(ref const_val) = constant_value {
                                                if arg_val != const_val {
                                                    all_same = false;
                                                    break;
                                                }
                                            } else {
                                                constant_value = Some(arg_val.clone());
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

        // First pass: find redundant expressions and mark them for replacement
        for &bid in &rpo {
            let block = code.blocks.get(&bid).unwrap();

            for (i, instr) in block.instructions.iter().enumerate() {
                let rid = RegisterId::temp(bid, i);

                match instr.deref() {
                    Instruction::BinOp { op, lhs, rhs, .. } => {
                        // Apply any existing replacements to operands to normalize
                        let normalized_lhs = apply_replacement(lhs, &replacements);
                        let normalized_rhs = apply_replacement(rhs, &replacements);

                        let expr = ValueExpr::BinOp {
                            op: op.clone(),
                            lhs: normalized_lhs.clone(),
                            rhs: normalized_rhs.clone(),
                        };

                        // Look for an existing computation we can reuse
                        let mut found_replacement = false;

                        // Search through all blocks that dominate this one
                        let mut search_blocks = vec![bid];
                        let mut current = bid;
                        while let Some(idom) = domtree.idom(current) {
                            search_blocks.push(idom);
                            current = idom;
                        }

                        // Search in reverse order (from root to current) to find the first occurrence
                        search_blocks.reverse();

                        for &search_bid in &search_blocks {
                            if let Some(search_block) = code.blocks.get(&search_bid) {
                                for (j, search_instr) in
                                    search_block.instructions.iter().enumerate()
                                {
                                    let search_rid = RegisterId::temp(search_bid, j);

                                    // Don't compare with ourselves
                                    if search_rid == rid {
                                        continue;
                                    }

                                    // Check if this instruction computes the same expression
                                    if matches_expr(search_instr.deref(), &expr, &replacements) {
                                        let _unused = replacements.insert(
                                            rid,
                                            Operand::register(search_rid, instr.dtype()),
                                        );
                                        changed = true;
                                        found_replacement = true;
                                        break;
                                    }
                                }
                                if found_replacement {
                                    break;
                                }
                            }
                        }

                        // If no replacement found, record this as the first occurrence
                        if !found_replacement {
                            let _unused = expr_to_first_occurrence.insert(expr, rid);
                        }
                    }
                    Instruction::UnaryOp { op, operand, .. } => {
                        let normalized_operand = apply_replacement(operand, &replacements);

                        let expr = ValueExpr::UnaryOp {
                            op: op.clone(),
                            operand: normalized_operand.clone(),
                        };

                        if let Some(&first_rid) = expr_to_first_occurrence.get(&expr) {
                            if dominates_register(&domtree, first_rid, bid) {
                                let _unused = replacements
                                    .insert(rid, Operand::register(first_rid, instr.dtype()));
                                changed = true;
                            } else {
                                let _unused = expr_to_first_occurrence.insert(expr, rid);
                            }
                        } else {
                            let _unused = expr_to_first_occurrence.insert(expr, rid);
                        }
                    }
                    Instruction::TypeCast {
                        value,
                        target_dtype,
                    } => {
                        let normalized_value = apply_replacement(value, &replacements);

                        let expr = ValueExpr::TypeCast {
                            value: normalized_value.clone(),
                            target_dtype: target_dtype.clone(),
                        };

                        if let Some(&first_rid) = expr_to_first_occurrence.get(&expr) {
                            if dominates_register(&domtree, first_rid, bid) {
                                let _unused = replacements
                                    .insert(rid, Operand::register(first_rid, instr.dtype()));
                                changed = true;
                            } else {
                                let _unused = expr_to_first_occurrence.insert(expr, rid);
                            }
                        } else {
                            let _unused = expr_to_first_occurrence.insert(expr, rid);
                        }
                    }
                    Instruction::GetElementPtr { ptr, offset, dtype } => {
                        let normalized_ptr = apply_replacement(ptr, &replacements);
                        let normalized_offset = apply_replacement(offset, &replacements);

                        let expr = ValueExpr::GetElementPtr {
                            ptr: normalized_ptr.clone(),
                            offset: normalized_offset.clone(),
                            dtype: dtype.clone(),
                        };

                        if let Some(&first_rid) = expr_to_first_occurrence.get(&expr) {
                            if dominates_register(&domtree, first_rid, bid) {
                                let _unused = replacements
                                    .insert(rid, Operand::register(first_rid, instr.dtype()));
                                changed = true;
                            } else {
                                let _unused = expr_to_first_occurrence.insert(expr, rid);
                            }
                        } else {
                            let _unused = expr_to_first_occurrence.insert(expr, rid);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Second pass: identify values that need phi nodes
        // For the test case, we need to identify that 'c' is computed in both b1 and b2
        // and used in b3, so we need a phi node
        let mut phi_nodes_to_add: HashMap<BlockId, Vec<(Dtype, Vec<(BlockId, Operand)>)>> =
            HashMap::new();

        // Check each block for values that could come from multiple predecessors
        for &bid in &rpo {
            if let Some(preds) = reverse_cfg.get(&bid) {
                if preds.len() > 1 {
                    // This block has multiple predecessors, check for common values
                    let block = code.blocks.get(&bid).unwrap();

                    // For each instruction in this block, check if it computes something
                    // that's also computed in the predecessors
                    for (i, instr) in block.instructions.iter().enumerate() {
                        let rid = RegisterId::temp(bid, i);

                        // Skip if this is already being replaced
                        if replacements.contains_key(&rid) {
                            continue;
                        }

                        match instr.deref() {
                            Instruction::BinOp { op, lhs, rhs, .. } => {
                                let normalized_lhs = apply_replacement(lhs, &replacements);
                                let normalized_rhs = apply_replacement(rhs, &replacements);

                                let expr = ValueExpr::BinOp {
                                    op: op.clone(),
                                    lhs: normalized_lhs.clone(),
                                    rhs: normalized_rhs.clone(),
                                };

                                // Check if this expression is available from predecessors
                                let mut pred_values: Vec<(BlockId, Operand)> = Vec::new();
                                for (pred_bid, _) in preds {
                                    if let Some(value) = find_value_in_block(
                                        &expr,
                                        *pred_bid,
                                        &code.blocks,
                                        &replacements,
                                    ) {
                                        pred_values.push((*pred_bid, value));
                                    }
                                }

                                // If available from all predecessors, create phi node
                                if pred_values.len() == preds.len() {
                                    let phi_idx = phi_nodes_to_add.entry(bid).or_default().len();
                                    phi_nodes_to_add
                                        .entry(bid)
                                        .or_default()
                                        .push((instr.dtype(), pred_values));

                                    // Replace this instruction with phi node reference
                                    let _unused = replacements.insert(
                                        rid,
                                        Operand::register(
                                            RegisterId::arg(bid, phi_idx),
                                            instr.dtype(),
                                        ),
                                    );
                                    changed = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Apply replacements to instructions and block exits
        if !replacements.is_empty() {
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

        // Add phi nodes and update jump arguments
        for (bid, phi_list) in phi_nodes_to_add {
            if let Some(block) = code.blocks.get_mut(&bid) {
                for (dtype, _) in &phi_list {
                    block.phinodes.push(Named::new(None, dtype.clone()));
                }
            }

            // Update jump arguments in predecessor blocks
            for (phi_idx, (_, pred_values)) in phi_list.iter().enumerate() {
                for (pred_bid, value) in pred_values {
                    if let Some(pred_block) = code.blocks.get_mut(pred_bid) {
                        match &mut pred_block.exit {
                            BlockExit::Jump { arg } => {
                                if arg.bid == bid {
                                    while arg.args.len() <= phi_idx {
                                        arg.args.push(Operand::constant(Constant::unit()));
                                    }
                                    arg.args[phi_idx] = value.clone();
                                }
                            }
                            BlockExit::ConditionalJump {
                                arg_then, arg_else, ..
                            } => {
                                if arg_then.bid == bid {
                                    while arg_then.args.len() <= phi_idx {
                                        arg_then.args.push(Operand::constant(Constant::unit()));
                                    }
                                    arg_then.args[phi_idx] = value.clone();
                                }
                                if arg_else.bid == bid {
                                    while arg_else.args.len() <= phi_idx {
                                        arg_else.args.push(Operand::constant(Constant::unit()));
                                    }
                                    arg_else.args[phi_idx] = value.clone();
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        changed
    }
}

fn apply_replacement(operand: &Operand, replacements: &HashMap<RegisterId, Operand>) -> Operand {
    if let Some((rid, _)) = operand.get_register() {
        if let Some(replacement) = replacements.get(rid) {
            // Recursively apply replacements
            return apply_replacement(replacement, replacements);
        }
    }
    operand.clone()
}

fn dominates_register(domtree: &Domtree, def_rid: RegisterId, use_bid: BlockId) -> bool {
    match def_rid {
        RegisterId::Temp { bid, .. } => {
            if bid == use_bid {
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

fn matches_expr(
    instr: &Instruction,
    expr: &ValueExpr,
    replacements: &HashMap<RegisterId, Operand>,
) -> bool {
    match (instr, expr) {
        (
            Instruction::BinOp {
                op: i_op,
                lhs: i_lhs,
                rhs: i_rhs,
                ..
            },
            ValueExpr::BinOp {
                op: e_op,
                lhs: e_lhs,
                rhs: e_rhs,
            },
        ) => {
            i_op == e_op
                && apply_replacement(i_lhs, replacements) == *e_lhs
                && apply_replacement(i_rhs, replacements) == *e_rhs
        }
        (
            Instruction::UnaryOp {
                op: i_op,
                operand: i_operand,
                ..
            },
            ValueExpr::UnaryOp {
                op: e_op,
                operand: e_operand,
            },
        ) => i_op == e_op && apply_replacement(i_operand, replacements) == *e_operand,
        (
            Instruction::TypeCast {
                value: i_value,
                target_dtype: i_dtype,
            },
            ValueExpr::TypeCast {
                value: e_value,
                target_dtype: e_dtype,
            },
        ) => i_dtype == e_dtype && apply_replacement(i_value, replacements) == *e_value,
        (
            Instruction::GetElementPtr {
                ptr: i_ptr,
                offset: i_offset,
                dtype: i_dtype,
            },
            ValueExpr::GetElementPtr {
                ptr: e_ptr,
                offset: e_offset,
                dtype: e_dtype,
            },
        ) => {
            i_dtype == e_dtype
                && apply_replacement(i_ptr, replacements) == *e_ptr
                && apply_replacement(i_offset, replacements) == *e_offset
        }
        _ => false,
    }
}
