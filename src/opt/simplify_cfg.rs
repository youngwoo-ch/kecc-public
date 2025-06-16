use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use itertools::izip;
use lang_c::visit;

use crate::ir::*;
use crate::opt::opt_utils::*;
use crate::opt::*;

pub type SimplifyCfg = FunctionPass<
    Repeat<(
        SimplifyCfgConstProp,
        (SimplifyCfgReach, (SimplifyCfgMerge, SimplifyCfgEmpty)),
    )>,
>;

/// Simplifies block exits by propagating constants.
#[derive(Default, Clone, Copy, Debug)]
pub struct SimplifyCfgConstProp {}

/// Retains only those blocks that are reachable from the init.
#[derive(Default, Clone, Copy, Debug)]
pub struct SimplifyCfgReach {}

/// Merges two blocks if a block is pointed to only by another
#[derive(Default, Clone, Copy, Debug)]
pub struct SimplifyCfgMerge {}

/// Removes empty blocks
#[derive(Default, Clone, Copy, Debug)]
pub struct SimplifyCfgEmpty {}

// fn make_cfg(fdef: &FunctionDefinition) -> HashMap<BlockId, Vec<JumpArg>> {
//     fdef.blocks
//         .iter()
//         .map(|(bid, block)| {
//             let mut args = Vec::new();
//             match &block.exit {
//                 BlockExit::Jump { arg } => args.push(arg.clone()),
//                 BlockExit::ConditionalJump {
//                     arg_then, arg_else, ..
//                 } => {
//                     args.push(arg_then.clone());
//                     args.push(arg_else.clone());
//                 }
//                 BlockExit::Switch { default, cases, .. } => {
//                     args.push(default.clone());
//                     for (_, arg) in cases {
//                         args.push(arg.clone());
//                     }
//                 }
//                 _ => {}
//             }
//             (*bid, args)
//         })
//         .collect()
// }

fn walk_function_operands<F>(fdef: &mut FunctionDefinition, mut f: F)
where
    F: FnMut(&mut Operand),
{
    for (_, block) in fdef.blocks.iter_mut() {
        for instruction in block.instructions.iter_mut() {
            walk_instruction_operands(instruction.deref_mut(), &mut f);
        }

        walk_block_exit_operands(&mut block.exit, &mut f);
    }
}

fn walk_instruction_operands<F>(instruction: &mut Instruction, f: &mut F)
where
    F: FnMut(&mut Operand),
{
    match instruction {
        Instruction::Nop => {}
        Instruction::Value { value } => f(value),
        Instruction::BinOp { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        Instruction::UnaryOp { operand, .. } => f(operand),
        Instruction::Store { ptr, value } => {
            f(ptr);
            f(value);
        }
        Instruction::Load { ptr } => f(ptr),
        Instruction::Call { callee, args, .. } => {
            f(callee);
            for arg in args {
                f(arg);
            }
        }
        Instruction::TypeCast { value, .. } => f(value),
        Instruction::GetElementPtr { ptr, offset, .. } => {
            f(ptr);
            f(offset);
        }
    }
}

fn walk_block_exit_operands<F>(exit: &mut BlockExit, f: &mut F)
where
    F: FnMut(&mut Operand),
{
    match exit {
        BlockExit::Jump { arg } => {
            for operand in &mut arg.args {
                f(operand);
            }
        }
        BlockExit::ConditionalJump {
            condition,
            arg_then,
            arg_else,
        } => {
            f(condition);
            for operand in &mut arg_then.args {
                f(operand);
            }
            for operand in &mut arg_else.args {
                f(operand);
            }
        }
        BlockExit::Switch {
            value,
            default,
            cases,
        } => {
            f(value);
            for operand in &mut default.args {
                f(operand);
            }
            for (_, arg) in cases {
                for operand in &mut arg.args {
                    f(operand);
                }
            }
        }
        BlockExit::Return { value } => f(value),
        BlockExit::Unreachable => {}
    }
}

fn replace_operands(operand: &mut Operand, replaces: &HashMap<RegisterId, Operand>) {
    if let Some((rid, _)) = operand.get_register() {
        if let Some(replacement) = replaces.get(rid) {
            *operand = replacement.clone();
        }
    }
}

impl Optimize<FunctionDefinition> for SimplifyCfgConstProp {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        code.blocks.iter_mut().any(|(_, block)| {
            if let Some(exit) = self.simplify_block_exit(&block.exit) {
                block.exit = exit;
                true
            } else {
                false
            }
        })
    }
}

impl Optimize<FunctionDefinition> for SimplifyCfgReach {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let graph = make_cfg(code);

        let mut queue = vec![code.bid_init];
        let mut visited = HashSet::new();
        let _unused = visited.insert(code.bid_init);
        while let Some(bid) = queue.pop() {
            if let Some(args) = graph.get(&bid) {
                for arg in args {
                    if visited.insert(arg.bid) {
                        queue.push(arg.bid);
                    }
                }
            }
        }

        let size_orig = code.blocks.len();
        code.blocks.retain(|bid, _| visited.contains(bid));

        code.blocks.len() < size_orig
    }
}

impl Optimize<FunctionDefinition> for SimplifyCfgMerge {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let graph = make_cfg(code);

        let mut indegrees = HashMap::<BlockId, usize>::new();
        for args in graph.values() {
            for arg in args {
                *indegrees.entry(arg.bid).or_insert(0) += 1;
            }
        }

        let mut merge_candidates = Vec::new();
        for (bid_from, block_from) in code.blocks.iter() {
            if let BlockExit::Jump { arg } = &block_from.exit {
                if *bid_from != arg.bid && indegrees.get(&arg.bid) == Some(&1) {
                    merge_candidates.push((*bid_from, arg.bid, arg.args.clone()));
                }
            }
        }

        if merge_candidates.is_empty() {
            return false;
        }

        let mut all_replaces = HashMap::new();

        for (bid_from, bid_to, args_to) in merge_candidates {
            if !code.blocks.contains_key(&bid_from) || !code.blocks.contains_key(&bid_to) {
                continue;
            }

            let block_to = code.blocks.remove(&bid_to).unwrap();
            let block_from = code.blocks.get_mut(&bid_from).unwrap();

            let mut replaces = HashMap::new();

            for (i, (a, p)) in izip!(&args_to, block_to.phinodes.iter()).enumerate() {
                assert_eq!(&a.dtype(), p.deref());
                let from = RegisterId::arg(bid_to, i);
                let _unused = replaces.insert(from, a.clone());
            }

            let len = block_from.instructions.len();
            for (i, instr) in block_to.instructions.into_iter().enumerate() {
                let dtype = instr.dtype();
                block_from.instructions.push(instr);
                let from = RegisterId::temp(bid_to, i);
                let to = Operand::register(RegisterId::temp(bid_from, len + i), dtype.clone());
                let _unused = replaces.insert(from, to);
            }

            block_from.exit = block_to.exit;

            for (k, v) in replaces {
                let _unused = all_replaces.insert(k, v);
            }
        }

        walk_function_operands(code, |operand| replace_operands(operand, &all_replaces));

        true
    }
}

impl Optimize<FunctionDefinition> for SimplifyCfgEmpty {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let empty_blocks = code
            .blocks
            .iter()
            .filter_map(|(bid, block)| {
                if block.phinodes.is_empty() && block.instructions.is_empty() {
                    Some((*bid, block.clone()))
                } else {
                    None
                }
            })
            .collect::<HashMap<_, _>>();

        code.blocks
            .iter_mut()
            .any(|(_, block)| self.simplify_block_exit(&mut block.exit, &empty_blocks))
    }
}

impl SimplifyCfgConstProp {
    fn simplify_block_exit(&self, exit: &BlockExit) -> Option<BlockExit> {
        match exit {
            BlockExit::ConditionalJump {
                condition,
                arg_then,
                arg_else,
            } => {
                if arg_then == arg_else {
                    return Some(BlockExit::Jump {
                        arg: arg_then.clone(),
                    });
                }

                if let Some(c) = condition.get_constant() {
                    match c {
                        Constant::Int { value: 0, .. } => {
                            return Some(BlockExit::Jump {
                                arg: arg_else.clone(),
                            });
                        }
                        Constant::Int { value: 1, .. } => {
                            return Some(BlockExit::Jump {
                                arg: arg_then.clone(),
                            });
                        }
                        _ => {}
                    }
                }

                None
            }
            BlockExit::Switch {
                value,
                default,
                cases,
            } => {
                if cases.iter().all(|(_, bid)| default == bid) {
                    return Some(BlockExit::Jump {
                        arg: default.clone(),
                    });
                }

                if let Some(v) = value.get_constant() {
                    return Some(BlockExit::Jump {
                        arg: if let Some((_, arg)) = cases.iter().find(|(c, _)| v == c) {
                            arg.clone()
                        } else {
                            default.clone()
                        },
                    });
                }

                None
            }
            _ => None,
        }
    }
}

impl SimplifyCfgEmpty {
    fn simplify_jump_arg(&self, arg: &mut JumpArg, empty_blocks: &HashMap<BlockId, Block>) -> bool {
        let Some(block) = empty_blocks.get(&arg.bid) else {
            return false;
        };

        assert!(arg.args.is_empty());

        if let BlockExit::Jump { arg: a } = &block.exit {
            *arg = a.clone();
            true
        } else {
            false
        }
    }

    fn simplify_block_exit(
        &self,
        exit: &mut BlockExit,
        empty_blocks: &HashMap<BlockId, Block>,
    ) -> bool {
        match exit {
            BlockExit::Jump { arg } => {
                let Some(block) = empty_blocks.get(&arg.bid) else {
                    return false;
                };
                *exit = block.exit.clone();
                true
            }
            BlockExit::ConditionalJump {
                arg_then, arg_else, ..
            } => {
                let changed1 = self.simplify_jump_arg(arg_then, empty_blocks);
                let changed2 = self.simplify_jump_arg(arg_else, empty_blocks);
                changed1 || changed2
            }
            BlockExit::Switch { default, cases, .. } => {
                let changed1 = self.simplify_jump_arg(default, empty_blocks);
                let changed2 = cases
                    .iter_mut()
                    .any(|c| self.simplify_jump_arg(&mut c.1, empty_blocks));
                changed1 || changed2
            }
            BlockExit::Return { .. } | BlockExit::Unreachable => false,
        }
    }
}
