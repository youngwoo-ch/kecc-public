//! Utilities for implementing optimizations.
//!
//! You can freely add utilities commonly used in the implementation of multiple optimizations here.

#![allow(dead_code)]

use core::cmp::Ordering;
use core::ops::{Deref, DerefMut};
use lang_c::visit;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::*;
use crate::opt::*;

macro_rules! some_or {
    ($expr:expr, $alt:expr) => {
        match $expr {
            Some(val) => val,
            None => $alt,
        }
    };
}

pub(crate) fn make_cfg(fdef: &FunctionDefinition) -> HashMap<BlockId, Vec<JumpArg>> {
    fdef.blocks
        .iter()
        .map(|(bid, block)| {
            let mut args = Vec::new();
            match &block.exit {
                BlockExit::Jump { arg } => args.push(arg.clone()),
                BlockExit::ConditionalJump {
                    arg_then, arg_else, ..
                } => {
                    args.push(arg_then.clone());
                    args.push(arg_else.clone());
                }
                BlockExit::Switch { default, cases, .. } => {
                    args.push(default.clone());
                    for (_, arg) in cases {
                        args.push(arg.clone());
                    }
                }
                _ => {}
            }
            (*bid, args)
        })
        .collect()
}

pub(crate) fn reverse_cfg(
    cfg: &HashMap<BlockId, Vec<JumpArg>>,
) -> HashMap<BlockId, Vec<(BlockId, JumpArg)>> {
    let mut result = HashMap::new();

    for (bid, jumps) in cfg {
        for jump in jumps {
            result
                .entry(jump.bid)
                .or_insert_with(Vec::new)
                .push((*bid, jump.clone()));
        }
    }

    result
}

pub(crate) trait Walk {
    fn walk<F>(&mut self, f: F) -> bool
    where
        F: FnMut(&mut Operand) -> bool;
}

impl Walk for FunctionDefinition {
    fn walk<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&mut Operand) -> bool,
    {
        self.blocks.iter_mut().any(|(_, block)| block.walk(&mut f))
    }
}

impl Walk for Block {
    fn walk<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&mut Operand) -> bool,
    {
        let changed1 = self.instructions.iter_mut().any(|i| i.walk(&mut f));
        let changed2 = self.exit.walk(&mut f);
        changed1 || changed2
    }
}

impl Walk for Instruction {
    fn walk<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&mut Operand) -> bool,
    {
        match self {
            Instruction::Nop => false,
            Instruction::Value { value } => f(value),
            Instruction::BinOp { lhs, rhs, .. } => {
                let changed1 = f(lhs);
                let changed2 = f(rhs);
                changed1 || changed2
            }
            Instruction::UnaryOp { operand, .. } => f(operand),
            Instruction::Store { ptr, value } => {
                let changed1 = f(ptr);
                let changed2 = f(value);
                changed1 || changed2
            }
            Instruction::Load { ptr } => f(ptr),
            Instruction::Call { callee, args, .. } => {
                let changed1 = f(callee);
                let changed2 = args.iter_mut().any(&mut f);
                changed1 || changed2
            }
            Instruction::TypeCast { value, .. } => f(value),
            Instruction::GetElementPtr { ptr, offset, .. } => {
                let changed1 = f(ptr);
                let changed2 = f(offset);
                changed1 || changed2
            }
        }
    }
}

impl Walk for BlockExit {
    fn walk<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&mut Operand) -> bool,
    {
        match self {
            BlockExit::Jump { arg } => arg.args.iter_mut().any(&mut f),
            BlockExit::ConditionalJump {
                condition,
                arg_then,
                arg_else,
            } => {
                let changed1 = f(condition);
                let changed2 = arg_then.args.iter_mut().any(&mut f);
                let changed3 = arg_else.args.iter_mut().any(&mut f);
                changed1 || changed2 || changed3
            }
            BlockExit::Switch {
                value,
                default,
                cases,
            } => {
                let changed1 = f(value);
                let changed2 = default.args.iter_mut().any(&mut f);
                let changed3 = cases
                    .iter_mut()
                    .any(|(_, arg)| arg.args.iter_mut().any(&mut f));
                changed1 || changed2 || changed3
            }
            BlockExit::Return { value } => f(value),
            BlockExit::Unreachable => false,
        }
    }
}

pub(crate) struct PostOrder<'a> {
    visited: HashSet<BlockId>,
    cfg: &'a HashMap<BlockId, Vec<JumpArg>>,
    traversed: Vec<BlockId>,
}

impl PostOrder<'_> {
    fn traverse(&mut self, bid: BlockId) {
        for jump in self.cfg.get(&bid).unwrap() {
            if self.visited.insert(jump.bid) {
                self.traverse(jump.bid);
            }
        }
        self.traversed.push(bid);
    }
}

pub(crate) fn traverse_postorder(
    bid_init: BlockId,
    cfg: &HashMap<BlockId, Vec<JumpArg>>,
) -> Vec<BlockId> {
    let mut post_order = PostOrder {
        visited: HashSet::new(),
        cfg,
        traversed: Vec::new(),
    };
    let _unused = post_order.visited.insert(bid_init);
    post_order.traverse(bid_init);
    post_order.traversed
}

#[derive(Debug)]
pub(crate) struct Domtree {
    idoms: HashMap<BlockId, BlockId>,
    frontiers: HashMap<BlockId, Vec<BlockId>>,
    reverse_post_order: Vec<BlockId>,
}

impl Domtree {
    pub(crate) fn new(
        bid_init: BlockId,
        cfg: &HashMap<BlockId, Vec<JumpArg>>,
        reverse_cfg: &HashMap<BlockId, Vec<(BlockId, JumpArg)>>,
    ) -> Self {
        let mut reverse_post_order = traverse_postorder(bid_init, cfg);
        reverse_post_order.reverse();

        let inverse_reverse_post_order = reverse_post_order
            .iter()
            .enumerate()
            .map(|(i, bid)| (*bid, i))
            .collect();

        let mut idoms = HashMap::<BlockId, BlockId>::new();

        loop {
            let mut changed = false;

            for bid in &reverse_post_order {
                if *bid == bid_init {
                    continue;
                }

                let mut idom = None;

                for (bid_prev, _) in reverse_cfg.get(bid).unwrap() {
                    if *bid_prev == bid_init || idoms.contains_key(bid_prev) {
                        idom = Some(intersect_idom(
                            idom,
                            *bid_prev,
                            &inverse_reverse_post_order,
                            &idoms,
                        ));
                    }
                }

                if let Some(idom) = idom {
                    let _unused = idoms
                        .entry(*bid)
                        .and_modify(|v| {
                            if *v != idom {
                                changed = true;
                                *v = idom;
                            }
                        })
                        .or_insert_with(|| {
                            changed = true;
                            idom
                        });
                }
            }

            if !changed {
                break;
            }
        }

        let mut frontiers = HashMap::new();

        for (bid, prevs) in reverse_cfg {
            if prevs.len() <= 1 {
                continue;
            }

            let idom = *some_or!(idoms.get(bid), continue);

            for (bid_prev, _) in prevs {
                let mut runner = *bid_prev;
                while !Self::dominates(&idoms, runner, *bid) {
                    frontiers.entry(runner).or_insert_with(Vec::new).push(*bid);
                    runner = *idoms.get(&runner).unwrap();
                }
            }
        }

        Self {
            idoms,
            frontiers,
            reverse_post_order,
        }
    }

    pub(crate) fn idom(&self, bid: BlockId) -> Option<BlockId> {
        self.idoms.get(&bid).cloned()
    }

    pub(crate) fn dominates(
        idoms: &HashMap<BlockId, BlockId>,
        bid1: BlockId,
        mut bid2: BlockId,
    ) -> bool {
        loop {
            bid2 = *some_or!(idoms.get(&bid2), return false);
            if bid1 == bid2 {
                return true;
            }
        }
    }

    pub(crate) fn frontier(&self, bid: BlockId) -> Option<&Vec<BlockId>> {
        self.frontiers.get(&bid)
    }

    pub(crate) fn walk<F>(&self, mut f: F)
    where
        F: FnMut(Option<BlockId>, BlockId),
    {
        for bid in &self.reverse_post_order {
            f(self.idoms.get(bid).cloned(), *bid);
        }
    }
}

pub(crate) fn intersect_idom(
    lhs: Option<BlockId>,
    mut rhs: BlockId,
    inverse_reverse_post_order: &HashMap<BlockId, usize>,
    idoms: &HashMap<BlockId, BlockId>,
) -> BlockId {
    let mut lhs = some_or!(lhs, return rhs);

    loop {
        if lhs == rhs {
            return lhs;
        }

        let lhs_index = inverse_reverse_post_order.get(&lhs).unwrap();
        let rhs_index = inverse_reverse_post_order.get(&rhs).unwrap();

        match lhs_index.cmp(rhs_index) {
            Ordering::Less => rhs = *idoms.get(&rhs).unwrap(),
            Ordering::Greater => lhs = *idoms.get(&lhs).unwrap(),
            Ordering::Equal => panic!("intersect_idom: lhs and rhs are the same block"),
        }
    }
}
