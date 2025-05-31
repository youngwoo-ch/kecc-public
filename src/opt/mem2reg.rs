use core::ops::{Deref, DerefMut};
use std::collections::{BTreeMap, HashMap, HashSet};

use lang_c::visit;

use crate::ir::*;
use crate::opt::opt_utils::*;
use crate::opt::*;

use core::cmp::Ordering;

macro_rules! some_or {
    ($expr:expr, $alt:expr) => {
        match $expr {
            Some(val) => val,
            None => $alt,
        }
    };
}

fn make_cfg(fdef: &FunctionDefinition) -> HashMap<BlockId, Vec<JumpArg>> {
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

fn reverse_cfg(cfg: &HashMap<BlockId, Vec<JumpArg>>) -> HashMap<BlockId, Vec<(BlockId, JumpArg)>> {
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

trait Walk {
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

struct PostOrder<'a> {
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

fn traverse_postorder(bid_init: BlockId, cfg: &HashMap<BlockId, Vec<JumpArg>>) -> Vec<BlockId> {
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
struct Domtree {
    idoms: HashMap<BlockId, BlockId>,
    frontiers: HashMap<BlockId, Vec<BlockId>>,
    reverse_post_order: Vec<BlockId>,
}

impl Domtree {
    fn new(
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

    fn idom(&self, bid: BlockId) -> Option<BlockId> {
        self.idoms.get(&bid).cloned()
    }

    fn dominates(idoms: &HashMap<BlockId, BlockId>, bid1: BlockId, mut bid2: BlockId) -> bool {
        loop {
            bid2 = *some_or!(idoms.get(&bid2), return false);
            if bid1 == bid2 {
                return true;
            }
        }
    }

    fn frontier(&self, bid: BlockId) -> Option<&Vec<BlockId>> {
        self.frontiers.get(&bid)
    }

    fn walk<F>(&self, mut f: F)
    where
        F: FnMut(Option<BlockId>, BlockId),
    {
        for bid in &self.reverse_post_order {
            f(self.idoms.get(bid).cloned(), *bid);
        }
    }
}

fn intersect_idom(
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

pub type Mem2reg = FunctionPass<Mem2regInner>;

#[derive(Default, Clone, Copy, Debug)]
pub struct Mem2regInner {}

#[derive(Clone, Debug, PartialEq)]
enum OperandVar {
    Operand(Operand),
    Phi((usize, BlockId)),
}

impl OperandVar {
    fn lookup(&self, dtype: &Dtype, phinode_indexes: &HashMap<(usize, BlockId), usize>) -> Operand {
        match self {
            OperandVar::Operand(operand) => operand.clone(),
            OperandVar::Phi((aid, bid)) => {
                let index = phinode_indexes.get(&(*aid, *bid)).unwrap_or_else(|| {
                    panic!("phi node must have been created for ({}, {})", aid, bid)
                });
                Operand::register(RegisterId::arg(*bid, *index), dtype.clone())
            }
        }
    }
}

struct JoinTable<'s> {
    inner: HashMap<(usize, BlockId), BlockId>,
    domtree: &'s Domtree,
    joins: &'s HashSet<(usize, BlockId)>,
}

impl<'s> JoinTable<'s> {
    fn new(domtree: &'s Domtree, joins: &'s HashSet<(usize, BlockId)>) -> Self {
        Self {
            inner: HashMap::new(),
            domtree,
            joins,
        }
    }

    fn lookup(&mut self, aid: usize, mut bid: BlockId) -> BlockId {
        let mut bids = Vec::new();
        let ret = loop {
            if let Some(ret) = self.inner.get(&(aid, bid)) {
                break *ret;
            }

            bids.push(bid);

            if self.joins.contains(&(aid, bid)) {
                break bid;
            }

            bid = some_or!(self.domtree.idom(bid), break bid);
        };

        for bid in bids {
            let _unused = self.inner.insert((aid, bid), ret);
        }

        ret
    }
}

fn mark_inpromotable(inpromotable: &mut HashSet<usize>, operand: &Operand) {
    if let Some((RegisterId::Local { aid }, _)) = operand.get_register() {
        let _unused = inpromotable.insert(*aid);
    }
}

impl Optimize<FunctionDefinition> for Mem2regInner {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let mut inpromotable = HashSet::new();
        let mut stores = HashMap::<usize, Vec<BlockId>>::new();

        for (bid, block) in &code.blocks {
            for instr in &block.instructions {
                match instr.deref() {
                    Instruction::Nop => {}
                    Instruction::BinOp { lhs, rhs, .. } => {
                        mark_inpromotable(&mut inpromotable, lhs);
                        mark_inpromotable(&mut inpromotable, rhs);
                    }
                    Instruction::UnaryOp { operand, .. } => {
                        mark_inpromotable(&mut inpromotable, operand);
                    }
                    Instruction::Store { ptr, value } => {
                        mark_inpromotable(&mut inpromotable, value);

                        let (rid, _) = some_or!(ptr.get_register(), continue);
                        if let RegisterId::Local { aid } = rid {
                            stores.entry(*aid).or_default().push(*bid);
                        }
                    }
                    Instruction::Load { .. } => {}
                    Instruction::Call { callee, args, .. } => {
                        mark_inpromotable(&mut inpromotable, callee);
                        for arg in args {
                            mark_inpromotable(&mut inpromotable, arg);
                        }
                    }
                    Instruction::TypeCast { value, .. } => {
                        mark_inpromotable(&mut inpromotable, value);
                    }
                    _ => {}
                }
            }
        }

        if (0..code.allocations.len()).all(|i| inpromotable.contains(&i)) {
            return false;
        }

        let cfg = make_cfg(code);
        let reverse_cfg = reverse_cfg(&cfg);
        let domtree = Domtree::new(code.bid_init, &cfg, &reverse_cfg);

        let joins: HashMap<usize, HashSet<BlockId>> = stores
            .iter()
            .filter(|(aid, _)| !inpromotable.contains(*aid))
            .map(|(aid, bids)| {
                (*aid, {
                    let mut stack = bids.clone();
                    let mut visited = HashSet::new();
                    while let Some(bid) = stack.pop() {
                        let bid_frontiers = some_or!(domtree.frontier(bid), continue);
                        for bid_frontier in bid_frontiers {
                            if visited.insert(*bid_frontier) {
                                stack.push(*bid_frontier);
                            }
                        }
                    }
                    visited
                })
            })
            .collect();

        let joins_set: HashSet<(usize, BlockId)> = joins
            .iter()
            .flat_map(|(aid, bids)| bids.iter().map(move |bid| (*aid, *bid)))
            .collect();
        let mut join_table = JoinTable::new(&domtree, &joins_set);
        let mut replaces = HashMap::<RegisterId, OperandVar>::new();
        let mut phinode_needed = HashSet::<(usize, BlockId)>::new();
        let mut end_values = HashMap::<(usize, BlockId), OperandVar>::new();

        for (bid, block) in &code.blocks {
            for (i, instr) in block.instructions.iter().enumerate() {
                match instr.deref() {
                    Instruction::Store { ptr, value } => {
                        let (rid, _) = some_or!(ptr.get_register(), continue);
                        if let RegisterId::Local { aid } = rid {
                            if inpromotable.contains(aid) {
                                continue;
                            }
                            let _unused =
                                end_values.insert((*aid, *bid), OperandVar::Operand(value.clone()));
                        }
                    }
                    Instruction::Load { ptr } => {
                        let (rid, _) = some_or!(ptr.get_register(), continue);
                        if let RegisterId::Local { aid } = rid {
                            if inpromotable.contains(aid) {
                                continue;
                            }

                            // Check if we already have a value for this allocation in this block
                            if let Some(var) = end_values.get(&(*aid, *bid)) {
                                let result =
                                    replaces.insert(RegisterId::temp(*bid, i), var.clone());
                                assert_eq!(result, None);
                                continue;
                            }

                            // Check if the current block will have a phi node for this allocation
                            let bid_join = join_table.lookup(*aid, *bid);
                            if bid_join == *bid && joins_set.contains(&(*aid, *bid)) {
                                // This block will have a phi node for this allocation
                                let var = OperandVar::Phi((*aid, *bid));
                                let _unused = end_values.insert((*aid, *bid), var.clone());
                                let result = replaces.insert(RegisterId::temp(*bid, i), var);
                                assert_eq!(result, None);
                                continue;
                            }

                            // Try to find the value at the join point
                            let var = if let Some(val) = end_values.get(&(*aid, bid_join)) {
                                val.clone()
                            } else {
                                // If we can't find a value at the join point, try to find it
                                // by walking up the dominator tree from the current block
                                let mut found = None;
                                let mut current = *bid;

                                // First check if the immediate dominator has a phi node for this allocation
                                if let Some(idom) = domtree.idom(current) {
                                    if joins_set.contains(&(*aid, idom)) {
                                        found = Some(OperandVar::Phi((*aid, idom)));
                                    }
                                }

                                // If not found, continue walking up the dominator tree
                                if found.is_none() {
                                    while let Some(idom) = domtree.idom(current) {
                                        if let Some(val) = end_values.get(&(*aid, idom)) {
                                            found = Some(val.clone());
                                            break;
                                        }
                                        if joins_set.contains(&(*aid, idom)) {
                                            found = Some(OperandVar::Phi((*aid, idom)));
                                            break;
                                        }
                                        current = idom;
                                    }
                                }

                                found.unwrap_or_else(|| {
                                    if bid_join == code.bid_init {
                                        let dtype = code.allocations.get(*aid).unwrap().deref();
                                        OperandVar::Operand(Operand::constant(Constant::undef(
                                            dtype.clone(),
                                        )))
                                    } else {
                                        let _unused = phinode_needed.insert((*aid, bid_join));
                                        OperandVar::Phi((*aid, bid_join))
                                    }
                                })
                            };

                            let _unused = end_values.insert((*aid, *bid), var.clone());
                            let result = replaces.insert(RegisterId::temp(*bid, i), var);
                            assert_eq!(result, None);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Track values at the end of each block more comprehensively
        let mut block_end_values: HashMap<(usize, BlockId), OperandVar> = end_values.clone();

        // Update with any stores in each block
        for (bid, block) in &code.blocks {
            for instr in &block.instructions {
                if let Instruction::Store { ptr, value } = instr.deref() {
                    if let Some((RegisterId::Local { aid }, _)) = ptr.get_register() {
                        if !inpromotable.contains(aid) {
                            // Check if the stored value references a load that will be replaced
                            let stored_var = if let Some((rid, _)) = value.get_register() {
                                if let Some(replacement) = replaces.get(rid) {
                                    replacement.clone()
                                } else {
                                    OperandVar::Operand(value.clone())
                                }
                            } else {
                                OperandVar::Operand(value.clone())
                            };
                            let _unused = block_end_values.insert((*aid, *bid), stored_var);
                        }
                    }
                }
            }
        }

        let mut phinode_visited = phinode_needed.clone();
        let mut phinode_stack = phinode_visited.iter().cloned().collect::<Vec<_>>();
        let mut phinodes =
            BTreeMap::<(usize, BlockId), (Dtype, HashMap<BlockId, OperandVar>)>::new();

        // Also add all join points to the stack
        for (aid, bid) in &joins_set {
            if phinode_visited.insert((*aid, *bid)) {
                phinode_stack.push((*aid, *bid));
            }
        }

        while let Some((aid, bid)) = phinode_stack.pop() {
            let mut cases = HashMap::new();
            let prevs = some_or!(reverse_cfg.get(&bid), continue);
            for (bid_prev, _) in prevs {
                let var_prev = (aid, *bid_prev);
                let end_value = end_values.get(&var_prev).cloned().unwrap_or_else(|| {
                    let bid_prev_phinode = join_table.lookup(aid, *bid_prev);
                    let var_prev_phinode = (aid, bid_prev_phinode);
                    if phinode_visited.insert(var_prev_phinode) {
                        phinode_stack.push(var_prev_phinode);
                    }
                    OperandVar::Phi(var_prev_phinode)
                });
                let _unused = cases.insert(*bid_prev, end_value);
            }

            let _unused = phinodes.insert(
                (aid, bid),
                (code.allocations.get(aid).unwrap().deref().clone(), cases),
            );
        }

        let mut phinode_indexes = HashMap::<(usize, BlockId), usize>::new();

        for ((aid, bid), (dtype, _)) in &phinodes {
            let name = code.allocations.get(*aid).unwrap().name();
            let block = code.blocks.get_mut(bid).unwrap();
            let index = block.phinodes.len();
            block
                .phinodes
                .push(Named::new(name.cloned(), dtype.clone()));
            let _unused = phinode_indexes.insert((*aid, *bid), index);
        }

        // Final update: Apply all replacements to block_end_values after we know about phi nodes
        let mut final_block_end_values = HashMap::new();
        for ((aid, bid), operand_var) in &block_end_values {
            let updated = match operand_var {
                OperandVar::Operand(operand) => {
                    if let Some((rid, dtype)) = operand.get_register() {
                        if let Some(replacement) = replaces.get(rid) {
                            // This operand references a load that will be replaced
                            replacement.lookup(dtype, &phinode_indexes)
                        } else {
                            operand.clone()
                        }
                    } else {
                        operand.clone()
                    }
                }
                OperandVar::Phi((phi_aid, phi_bid)) => {
                    let index = phinode_indexes
                        .get(&(*phi_aid, *phi_bid))
                        .expect("phi node must exist");
                    let dtype = code.allocations.get(*phi_aid).unwrap().deref();
                    Operand::register(RegisterId::arg(*phi_bid, *index), dtype.clone())
                }
            };
            let _unused = final_block_end_values.insert((*aid, *bid), OperandVar::Operand(updated));
        }

        // Also update the replaces map to use fully resolved values
        let mut final_replaces = HashMap::new();
        for (rid, operand_var) in &replaces {
            let resolved = match operand_var {
                OperandVar::Operand(operand) => {
                    if let Some((inner_rid, dtype)) = operand.get_register() {
                        if let Some(inner_replacement) = replaces.get(inner_rid) {
                            // This operand references another register that will be replaced
                            inner_replacement.clone()
                        } else {
                            operand_var.clone()
                        }
                    } else {
                        operand_var.clone()
                    }
                }
                _ => operand_var.clone(),
            };
            let _unused = final_replaces.insert(*rid, resolved);
        }
        let replaces = final_replaces;

        // Manual operand replacement to work around potential Walk trait issues with Named<Instruction>
        for block in code.blocks.values_mut() {
            // Replace operands in instructions
            for instr in block.instructions.iter_mut() {
                match instr.deref_mut() {
                    Instruction::BinOp { lhs, rhs, .. } => {
                        if let Some((rid, dtype)) = lhs.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *lhs = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                        if let Some((rid, dtype)) = rhs.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *rhs = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                    }
                    Instruction::UnaryOp { operand, .. } => {
                        if let Some((rid, dtype)) = operand.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *operand = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                    }
                    Instruction::Store { ptr, value } => {
                        if let Some((rid, dtype)) = ptr.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *ptr = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                        if let Some((rid, dtype)) = value.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *value = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                    }
                    Instruction::Load { ptr } => {
                        if let Some((rid, dtype)) = ptr.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *ptr = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                    }
                    Instruction::Call { callee, args, .. } => {
                        if let Some((rid, dtype)) = callee.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *callee = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                        for arg in args {
                            if let Some((rid, dtype)) = arg.get_register() {
                                if let Some(operand_var) = replaces.get(rid) {
                                    *arg = operand_var.lookup(dtype, &phinode_indexes);
                                }
                            }
                        }
                    }
                    Instruction::TypeCast { value, .. } => {
                        if let Some((rid, dtype)) = value.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *value = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                    }
                    Instruction::GetElementPtr { ptr, offset, .. } => {
                        if let Some((rid, dtype)) = ptr.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *ptr = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                        if let Some((rid, dtype)) = offset.get_register() {
                            if let Some(operand_var) = replaces.get(rid) {
                                *offset = operand_var.lookup(dtype, &phinode_indexes);
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Replace operands in block exit
            match &mut block.exit {
                BlockExit::Jump { .. } => {}
                BlockExit::ConditionalJump { condition, .. } => {
                    if let Some((rid, dtype)) = condition.get_register() {
                        if let Some(operand_var) = replaces.get(rid) {
                            *condition = operand_var.lookup(dtype, &phinode_indexes);
                        }
                    }
                }
                BlockExit::Switch { value, .. } => {
                    if let Some((rid, dtype)) = value.get_register() {
                        if let Some(operand_var) = replaces.get(rid) {
                            *value = operand_var.lookup(dtype, &phinode_indexes);
                        }
                    }
                }
                BlockExit::Return { value } => {
                    if let Some((rid, dtype)) = value.get_register() {
                        if let Some(operand_var) = replaces.get(rid) {
                            *value = operand_var.lookup(dtype, &phinode_indexes);
                        }
                    }
                }
                BlockExit::Unreachable => {}
            }
        }

        // Update jump arguments for phi nodes
        // First, we need to know which allocations map to which phi node positions
        let mut allocation_to_phi_index: HashMap<(usize, BlockId), usize> = HashMap::new();
        for (bid, block) in &code.blocks {
            for (i, phi) in block.phinodes.iter().enumerate() {
                // Find which allocation this phi node corresponds to
                for (aid, alloc) in code.allocations.iter().enumerate() {
                    if alloc.name() == phi.name() {
                        let _unused = allocation_to_phi_index.insert((aid, *bid), i);
                        break;
                    }
                }
            }
        }

        // Now update jump arguments based on the phinodes we created
        for ((aid, bid), (_, cases)) in &phinodes {
            // Get the phi node index for this allocation/block
            let phi_index = *allocation_to_phi_index
                .get(&(*aid, *bid))
                .or_else(|| phinode_indexes.get(&(*aid, *bid)))
                .expect("phi node index must exist");

            // Find all jumps to this block and update their arguments
            for (bid_from, operand_var) in cases {
                let block_from = code.blocks.get_mut(bid_from).unwrap();
                block_from.exit.walk_jump_args(|jump_arg| {
                    if jump_arg.bid == *bid {
                        // Find or add the argument for this phi node
                        let dtype = code.allocations.get(*aid).unwrap().deref();
                        let operand = match operand_var {
                            OperandVar::Operand(op) => op.clone(),
                            OperandVar::Phi((phi_aid, phi_bid)) => {
                                // Check if this phi node actually exists
                                if let Some(phi_index) = phinode_indexes.get(&(*phi_aid, *phi_bid))
                                {
                                    Operand::register(
                                        RegisterId::arg(*phi_bid, *phi_index),
                                        dtype.clone(),
                                    )
                                } else {
                                    // If the phi node doesn't exist (e.g., for init block), use undef
                                    Operand::constant(Constant::undef(dtype.clone()))
                                }
                            }
                        };

                        // Ensure we have enough args - resize if needed
                        while jump_arg.args.len() <= phi_index {
                            jump_arg.args.push(Operand::constant(Constant::unit()));
                        }
                        jump_arg.args[phi_index] = operand;
                    }
                });
            }
        }

        // Also need to ensure that all jumps have the correct number of arguments
        // First collect the expected number of arguments for each block
        let block_phinode_counts: HashMap<BlockId, usize> = code
            .blocks
            .iter()
            .map(|(bid, block)| (*bid, block.phinodes.len()))
            .collect();

        for (_, block) in code.blocks.iter_mut() {
            block.exit.walk_jump_args(|jump_arg| {
                if let Some(&expected_args) = block_phinode_counts.get(&jump_arg.bid) {
                    while jump_arg.args.len() < expected_args {
                        // This shouldn't happen if phi nodes are set up correctly,
                        // but add a placeholder just in case
                        jump_arg.args.push(Operand::constant(Constant::unit()));
                    }
                }
            });
        }

        // Final pass: For each jump, ensure arguments match the values that should flow
        // to the phi nodes in the target block
        // First collect information about phi nodes in each block
        let block_phi_info: HashMap<BlockId, Vec<(usize, String, Dtype)>> = code
            .blocks
            .iter()
            .map(|(bid, block)| {
                let phi_info = block
                    .phinodes
                    .iter()
                    .enumerate()
                    .map(|(i, phi)| {
                        let aid = code
                            .allocations
                            .iter()
                            .position(|a| a.name() == phi.name())
                            .unwrap_or(usize::MAX);
                        (
                            aid,
                            phi.name().cloned().unwrap_or_default(),
                            phi.deref().clone(),
                        )
                    })
                    .collect();
                (*bid, phi_info)
            })
            .collect();

        // Also collect phi node information for each block to avoid borrow checker issues
        let block_phi_names: HashMap<BlockId, Vec<Option<String>>> = code
            .blocks
            .iter()
            .map(|(bid, block)| {
                (
                    *bid,
                    block.phinodes.iter().map(|p| p.name().cloned()).collect(),
                )
            })
            .collect();

        // Update final_block_end_values with any stores that happen in blocks
        // This ensures we have the most up-to-date values when determining jump arguments
        for (bid, block) in &code.blocks {
            let mut current_values = HashMap::new();

            // Start with values at block entry
            for aid in 0..code.allocations.len() {
                if let Some(operand_var) = final_block_end_values.get(&(aid, *bid)) {
                    let _unused = current_values.insert(aid, operand_var.clone());
                }
            }

            // Update with stores in the block
            for instr in &block.instructions {
                if let Instruction::Store { ptr, value } = instr.deref() {
                    if let Some((RegisterId::Local { aid }, _)) = ptr.get_register() {
                        if !inpromotable.contains(aid) {
                            let _unused =
                                current_values.insert(*aid, OperandVar::Operand(value.clone()));
                        }
                    }
                }
            }

            // Update final_block_end_values with the values at block exit
            for (aid, operand_var) in current_values {
                let _unused = final_block_end_values.insert((aid, *bid), operand_var);
            }
        }

        for (bid_from, block_from) in code.blocks.iter_mut() {
            let bid_from = *bid_from;
            block_from.exit.walk_jump_args(|jump_arg| {
                let bid_to = jump_arg.bid;
                if let Some(phi_info) = block_phi_info.get(&bid_to) {
                    // Ensure we have the correct number of arguments
                    while jump_arg.args.len() < phi_info.len() {
                        jump_arg.args.push(Operand::constant(Constant::unit()));
                    }

                    // For each phi node in the target block, determine what value should flow from bid_from
                    for (i, (aid, phi_name, dtype)) in phi_info.iter().enumerate() {
                        if *aid != usize::MAX {
                            // Look up what value this allocation has at the end of bid_from
                            // If not found, walk up the dominator tree
                            let mut value = None;
                            let mut current_bid = bid_from;

                            loop {
                                if let Some(operand_var) =
                                    final_block_end_values.get(&(*aid, current_bid))
                                {
                                    value = Some(operand_var.lookup(dtype, &phinode_indexes));
                                    break;
                                }

                                // Check if there's a phi node in the current block
                                if let Some(phi_idx) =
                                    block_phi_names.get(&current_bid).and_then(|names| {
                                        names.iter().position(|n| n.as_ref() == Some(phi_name))
                                    })
                                {
                                    value = Some(Operand::register(
                                        RegisterId::arg(current_bid, phi_idx),
                                        dtype.clone(),
                                    ));
                                    break;
                                }

                                // Walk up the dominator tree
                                current_bid = match domtree.idom(current_bid) {
                                    Some(idom) => idom,
                                    None => break,
                                };
                            }

                            jump_arg.args[i] = value.unwrap_or_else(|| {
                                // No value found, use undef
                                Operand::constant(Constant::undef(dtype.clone()))
                            });
                        }
                    }

                    // Trim any extra arguments
                    jump_arg.args.truncate(phi_info.len());
                }
            });
        }

        for block in code.blocks.values_mut() {
            for instr in block.instructions.iter_mut() {
                match instr.deref().deref() {
                    Instruction::Store { ptr, .. } => {
                        let (rid, _) = some_or!(ptr.get_register(), continue);
                        if let RegisterId::Local { aid } = rid {
                            if !inpromotable.contains(aid) {
                                *instr.deref_mut() = Instruction::Nop;
                            }
                        }
                    }
                    Instruction::Load { ptr } => {
                        let (rid, _) = some_or!(ptr.get_register(), continue);
                        if let RegisterId::Local { aid } = rid {
                            if !inpromotable.contains(aid) {
                                *instr.deref_mut() = Instruction::Nop;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        true
    }
}
