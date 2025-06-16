use core::ops::Deref;
use std::collections::{BTreeMap, HashMap, HashSet};

use itertools::izip;
use lang_c::ast;

use crate::ir::*;
use crate::opt::opt_utils::*;
use crate::opt::*;

pub type Gvn = FunctionPass<GvnInner>;

type PhiNodesToAdd = HashMap<BlockId, Vec<(Dtype, Vec<(BlockId, Operand)>)>>;

#[derive(Default, Clone, Copy, Debug)]
pub struct GvnInner {}

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
    fn normalize(self) -> Self {
        match self {
            ValueExpr::BinOp { op, lhs, rhs } => match op {
                ast::BinaryOperator::Plus
                | ast::BinaryOperator::Multiply
                | ast::BinaryOperator::BitwiseAnd
                | ast::BinaryOperator::BitwiseXor
                | ast::BinaryOperator::BitwiseOr
                | ast::BinaryOperator::Equals
                | ast::BinaryOperator::NotEquals => {
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
            },
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

        let mut replacements: HashMap<RegisterId, Operand> = HashMap::new();

        let mut expr_to_first_occurrence: HashMap<ValueExpr, RegisterId> = HashMap::new();

        let post_order = traverse_postorder(code.bid_init, &cfg);
        let mut rpo = post_order;
        rpo.reverse();

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

            changed_this_round |= find_redundant_expressions(
                &rpo,
                &code.blocks,
                &domtree,
                &mut replacements,
                &mut expr_to_first_occurrence,
            );

            let phi_nodes_to_add = generate_phi_nodes_for_expressions(
                &rpo,
                &code.blocks,
                &reverse_cfg,
                &domtree,
                &mut replacements,
                &mut changed_this_round,
            );

            if !phi_nodes_to_add.is_empty() {
                add_phi_nodes(code, phi_nodes_to_add);
                changed_this_round = true;
            }

            if !changed_this_round || iterations >= MAX_ITERATIONS {
                break;
            }

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
            for (phi_idx, _phi) in block.phinodes.iter().enumerate() {
                if let Some(preds) = reverse_cfg.get(&bid) {
                    let mut constant_value: Option<Operand> = None;
                    let mut all_same = true;

                    for (pred_bid, _) in preds {
                        if let Some(pred_block) = blocks.get(pred_bid) {
                            match &pred_block.exit {
                                BlockExit::Jump { arg } => {
                                    if arg.bid == bid && phi_idx < arg.args.len() {
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
                                    if default.bid == bid && phi_idx < default.args.len() {
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

                                    for (_, case_arg) in cases {
                                        if case_arg.bid == bid && phi_idx < case_arg.args.len() {
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
            if !visited.insert(*rid) {
                return operand.clone();
            }
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
                true
            } else {
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
        _ => true,
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

        if let Some(replacement) = replacements.get(&rid) {
            if let Some((
                RegisterId::Temp {
                    bid: rep_bid,
                    iid: rep_iid,
                },
                _dtype,
            )) = replacement.get_register()
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
    if let Some(value) = find_value_in_block(expr, bid, blocks, replacements) {
        return Some(value);
    }

    let mut current = bid;
    while let Some(idom) = domtree.idom(current) {
        if let Some(value) = find_value_in_block(expr, idom, blocks, replacements) {
            if let Some((rid, _)) = value.get_register() {
                if dominates_register(domtree, *rid, bid) {
                    return Some(value);
                }
            } else {
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
        Instruction::Load { .. } => None,
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

            let expr_opt = instruction_to_expr(instr.deref(), replacements);

            if let Some(expr) = expr_opt {
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
        Instruction::Load { .. } => None,
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

    let mut search_blocks = vec![current_bid];
    let mut current = current_bid;
    while let Some(idom) = domtree.idom(current) {
        search_blocks.push(idom);
        current = idom;
    }

    search_blocks.reverse();

    for &search_bid in &search_blocks {
        if let Some(search_block) = blocks.get(&search_bid) {
            for (j, search_instr) in search_block.instructions.iter().enumerate() {
                let search_rid = RegisterId::temp(search_bid, j);

                if search_rid == current_rid {
                    continue;
                }

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

                if matches_expr(search_instr.deref(), expr, replacements) {
                    return Some(Operand::register(search_rid, dtype));
                }
            }
        }
    }

    None
}

fn matches_phi_from_block(operand: &Operand, bid: BlockId) -> bool {
    if let Some((RegisterId::Arg { bid: arg_bid, .. }, _)) = operand.get_register() {
        *arg_bid == bid
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
) -> PhiNodesToAdd {
    let mut phi_nodes_to_add: PhiNodesToAdd = HashMap::new();

    for &bid in rpo {
        if let Some(preds) = reverse_cfg.get(&bid) {
            if preds.len() > 1 {
                let block = blocks.get(&bid).unwrap();

                for (i, instr) in block.instructions.iter().enumerate() {
                    let rid = RegisterId::temp(bid, i);

                    if replacements.contains_key(&rid) {
                        continue;
                    }

                    let expr_opt = instruction_to_expr(instr.deref(), replacements);

                    if let Some(expr) = expr_opt {
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

                                if let Some(ref fv) = first_value {
                                    if &value != fv {
                                        all_same_value = false;
                                    }
                                } else {
                                    first_value = Some(value);
                                }
                            }
                        }

                        if pred_values.len() == preds.len() {
                            if all_same_value && first_value.is_some() {
                                let value = first_value.unwrap();
                                if let Some((value_rid, _)) = value.get_register() {
                                    if value_rid != &rid {
                                        let _unused = replacements.insert(rid, value);
                                        *changed = true;
                                    }
                                } else {
                                    let _unused = replacements.insert(rid, value);
                                    *changed = true;
                                }
                            } else {
                                let existing_phi_count = block.phinodes.len();
                                let relative_phi_idx =
                                    phi_nodes_to_add.entry(bid).or_default().len();
                                let absolute_phi_idx = existing_phi_count + relative_phi_idx;

                                phi_nodes_to_add
                                    .entry(bid)
                                    .or_default()
                                    .push((instr.dtype(), pred_values));

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

fn add_phi_nodes(code: &mut FunctionDefinition, phi_nodes_to_add: PhiNodesToAdd) {
    let original_phi_counts: HashMap<BlockId, usize> = phi_nodes_to_add
        .keys()
        .map(|bid| (*bid, code.blocks.get(bid).unwrap().phinodes.len()))
        .collect();

    for (bid, phi_list) in &phi_nodes_to_add {
        if let Some(block) = code.blocks.get_mut(bid) {
            for (dtype, _) in phi_list {
                block.phinodes.push(Named::new(None, dtype.clone()));
            }
        }

        let existing_phi_count = *original_phi_counts.get(bid).unwrap();
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
            if default.bid == target_bid {
                while default.args.len() <= phi_idx {
                    default.args.push(Operand::constant(Constant::unit()));
                }
                default.args[phi_idx] = value.clone();
            }

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
