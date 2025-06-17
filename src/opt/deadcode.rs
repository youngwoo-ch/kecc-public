use core::ops::Deref;
use std::collections::{HashMap, HashSet};

use crate::ir::*;
use crate::opt::opt_utils::*;
use crate::opt::*;

pub type Deadcode = FunctionPass<Repeat<DeadcodeInner>>;

#[derive(Default, Clone, Copy, Debug)]
pub struct DeadcodeInner {}

impl Optimize<FunctionDefinition> for DeadcodeInner {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let mut changed = false;

        let live_values = find_live_values(code);

        changed |= remove_dead_instructions(code, &live_values);
        changed |= remove_dead_phinodes(code, &live_values);
        changed |= remove_dead_allocations(code);

        changed
    }
}

fn find_live_values(code: &FunctionDefinition) -> HashSet<RegisterId> {
    let mut live = HashSet::new();
    let mut worklist = Vec::new();

    for (bid, block) in &code.blocks {
        match &block.exit {
            BlockExit::Return { value } => {
                if let Some((rid, _)) = value.get_register() {
                    worklist.push(*rid);
                }
            }
            BlockExit::ConditionalJump { condition, .. } => {
                if let Some((rid, _)) = condition.get_register() {
                    worklist.push(*rid);
                }
            }
            BlockExit::Switch { value, .. } => {
                if let Some((rid, _)) = value.get_register() {
                    worklist.push(*rid);
                }
            }
            _ => {}
        }

        let cfg = make_cfg(code);
        if let Some(jumps) = cfg.get(bid) {
            for jump in jumps {
                for arg in &jump.args {
                    if let Some((rid, _)) = arg.get_register() {
                        worklist.push(*rid);
                    }
                }
            }
        }

        for (i, instr) in block.instructions.iter().enumerate() {
            match instr.deref() {
                Instruction::Store { ptr, value } => {
                    if let Some((rid, _)) = ptr.get_register() {
                        worklist.push(*rid);
                    }
                    if let Some((rid, _)) = value.get_register() {
                        worklist.push(*rid);
                    }
                }
                Instruction::Call { callee, args, .. } => {
                    if let Some((rid, _)) = callee.get_register() {
                        worklist.push(*rid);
                    }
                    for arg in args {
                        if let Some((rid, _)) = arg.get_register() {
                            worklist.push(*rid);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    while let Some(rid) = worklist.pop() {
        if !live.insert(rid) {
            continue;
        }

        if let RegisterId::Temp { bid, iid } = rid {
            if let Some(block) = code.blocks.get(&bid) {
                if let Some(instr) = block.instructions.get(iid) {
                    match instr.deref() {
                        Instruction::Value { value } => {
                            if let Some((rid, _)) = value.get_register() {
                                worklist.push(*rid);
                            }
                        }
                        Instruction::BinOp { lhs, rhs, .. } => {
                            if let Some((rid, _)) = lhs.get_register() {
                                worklist.push(*rid);
                            }
                            if let Some((rid, _)) = rhs.get_register() {
                                worklist.push(*rid);
                            }
                        }
                        Instruction::UnaryOp { operand, .. } => {
                            if let Some((rid, _)) = operand.get_register() {
                                worklist.push(*rid);
                            }
                        }
                        Instruction::Load { ptr } => {
                            if let Some((rid, _)) = ptr.get_register() {
                                worklist.push(*rid);
                            }
                        }
                        Instruction::TypeCast { value, .. } => {
                            if let Some((rid, _)) = value.get_register() {
                                worklist.push(*rid);
                            }
                        }
                        Instruction::GetElementPtr { ptr, offset, .. } => {
                            if let Some((rid, _)) = ptr.get_register() {
                                worklist.push(*rid);
                            }
                            if let Some((rid, _)) = offset.get_register() {
                                worklist.push(*rid);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    live
}

fn remove_dead_instructions(
    code: &mut FunctionDefinition,
    live_values: &HashSet<RegisterId>,
) -> bool {
    let mut changed = false;
    let mut remapping = HashMap::new();

    for (bid, block) in &mut code.blocks {
        let mut new_instructions = Vec::new();
        let mut new_index = 0;

        for (old_index, instr) in block.instructions.iter().enumerate() {
            let rid = RegisterId::temp(*bid, old_index);

            let should_keep = match instr.deref() {
                Instruction::Nop => false,
                Instruction::Store { .. } | Instruction::Call { .. } => true,
                _ => live_values.contains(&rid),
            };

            if should_keep {
                if matches!(instr.deref(), Instruction::Nop) {
                    let _unused = remapping.insert(rid, Operand::constant(Constant::unit()));
                } else {
                    new_instructions.push(instr.clone());
                    if old_index != new_index {
                        let new_rid = RegisterId::temp(*bid, new_index);
                        let _unused =
                            remapping.insert(rid, Operand::register(new_rid, instr.dtype()));
                    }
                    new_index += 1;
                }
            } else {
                changed = true;
                if matches!(instr.dtype(), Dtype::Unit { .. }) {
                    let _unused = remapping.insert(rid, Operand::constant(Constant::unit()));
                }
            }
        }

        block.instructions = new_instructions;
    }

    if !remapping.is_empty() {
        let block_ids: Vec<BlockId> = code.blocks.keys().cloned().collect();

        for bid in block_ids {
            if let Some(block) = code.blocks.get_mut(&bid) {
                for instr in &mut block.instructions {
                    let _unused = instr.walk(|operand| {
                        if let Some((rid, _)) = operand.get_register() {
                            if let Some(new_operand) = remapping.get(rid) {
                                *operand = new_operand.clone();
                            }
                        }
                        false
                    });
                }

                let _unused = block.exit.walk(|operand| {
                    if let Some((rid, _)) = operand.get_register() {
                        if let Some(new_operand) = remapping.get(rid) {
                            *operand = new_operand.clone();
                        }
                    }
                    false
                });
            }
        }
    }

    changed
}

fn remove_dead_phinodes(code: &mut FunctionDefinition, live_values: &HashSet<RegisterId>) -> bool {
    let mut changed = false;
    let mut phi_remapping = HashMap::new();
    let mut blocks_phi_indices: HashMap<BlockId, Vec<usize>> = HashMap::new();

    for (bid, block) in &mut code.blocks {
        let mut new_phinodes = Vec::new();
        let mut phi_indices_to_keep = Vec::new();

        for (phi_idx, phi) in block.phinodes.iter().enumerate() {
            let rid = RegisterId::arg(*bid, phi_idx);
            if *bid == code.bid_init {
                new_phinodes.push(phi.clone());
                phi_indices_to_keep.push(phi_idx);
            } else {
                if live_values.contains(&rid) {
                    new_phinodes.push(phi.clone());
                    phi_indices_to_keep.push(phi_idx);
                } else {
                    changed = true;
                }
            }
        }

        for (new_idx, old_idx) in phi_indices_to_keep.iter().enumerate() {
            if new_idx != *old_idx {
                let _unused = phi_remapping.insert(
                    RegisterId::arg(*bid, *old_idx),
                    Operand::register(
                        RegisterId::arg(*bid, new_idx),
                        block.phinodes[*old_idx].deref().clone(),
                    ),
                );
            }
        }

        if new_phinodes.len() != block.phinodes.len() {
            let _unused = blocks_phi_indices.insert(*bid, phi_indices_to_keep);
        }

        block.phinodes = new_phinodes;
    }

    if !phi_remapping.is_empty() {
        for block in code.blocks.values_mut() {
            for instr in &mut block.instructions {
                let _unused = instr.walk(|operand| {
                    if let Some((rid, _)) = operand.get_register() {
                        if let Some(new_operand) = phi_remapping.get(rid) {
                            *operand = new_operand.clone();
                        }
                    }
                    false
                });
            }

            let _unused = block.exit.walk(|operand| {
                if let Some((rid, _)) = operand.get_register() {
                    if let Some(new_operand) = phi_remapping.get(rid) {
                        *operand = new_operand.clone();
                    }
                }
                false
            });
        }
    }

    if !blocks_phi_indices.is_empty() {
        let block_ids: Vec<BlockId> = code.blocks.keys().cloned().collect();
        for bid in block_ids {
            if let Some(block) = code.blocks.get_mut(&bid) {
                match &mut block.exit {
                    BlockExit::Jump { arg } => {
                        update_jump_arg_for_dead_phis(arg, &blocks_phi_indices);
                    }
                    BlockExit::ConditionalJump {
                        arg_then, arg_else, ..
                    } => {
                        update_jump_arg_for_dead_phis(arg_then, &blocks_phi_indices);
                        update_jump_arg_for_dead_phis(arg_else, &blocks_phi_indices);
                    }
                    BlockExit::Switch { default, cases, .. } => {
                        update_jump_arg_for_dead_phis(default, &blocks_phi_indices);
                        for (_, case_arg) in cases {
                            update_jump_arg_for_dead_phis(case_arg, &blocks_phi_indices);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    changed
}

fn update_jump_arg_for_dead_phis(
    jump_arg: &mut JumpArg,
    blocks_phi_indices: &HashMap<BlockId, Vec<usize>>,
) {
    if let Some(indices_to_keep) = blocks_phi_indices.get(&jump_arg.bid) {
        let mut new_args = Vec::new();
        for (idx, arg) in jump_arg.args.iter().enumerate() {
            if indices_to_keep.contains(&idx) {
                new_args.push(arg.clone());
            }
        }
        jump_arg.args = new_args;
    }
}

fn remove_dead_allocations(code: &mut FunctionDefinition) -> bool {
    let mut used_allocations = HashSet::new();

    let _unused = code.walk(|operand| {
        if let Some((RegisterId::Local { aid }, _)) = operand.get_register() {
            let _unused = used_allocations.insert(*aid);
        }
        false
    });

    let mut new_allocations = Vec::new();
    let mut remapping = HashMap::new();
    let mut new_index = 0;

    for (old_index, alloc) in code.allocations.iter().enumerate() {
        if used_allocations.contains(&old_index) {
            new_allocations.push(alloc.clone());
            if old_index != new_index {
                let _unused = remapping.insert(
                    RegisterId::local(old_index),
                    Operand::register(
                        RegisterId::local(new_index),
                        Dtype::pointer(alloc.deref().clone()),
                    ),
                );
            }
            new_index += 1;
        }
    }

    let changed = new_allocations.len() != code.allocations.len();
    code.allocations = new_allocations;

    if !remapping.is_empty() {
        let mut apply_remapping = |operand: &mut Operand| -> bool {
            if let Some((rid, _)) = operand.get_register() {
                if let RegisterId::Local { .. } = rid {
                    if let Some(new_operand) = remapping.get(rid) {
                        *operand = new_operand.clone();
                    }
                }
            }
            false
        };

        for block in code.blocks.values_mut() {
            for instr in &mut block.instructions {
                let _unused = instr.walk(&mut apply_remapping);
            }

            let _unused = block.exit.walk(&mut apply_remapping);
        }
    }

    changed
}
