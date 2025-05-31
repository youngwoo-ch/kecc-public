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
        // Phase 1: Analysis - determine which allocations can be promoted
        let (inpromotable, stores) = self.analyze_allocations(code);

        if (0..code.allocations.len()).all(|i| inpromotable.contains(&i)) {
            return false;
        }

        // Phase 2: Build control flow and dominance structures
        let cfg = make_cfg(code);
        let reverse_cfg = reverse_cfg(&cfg);
        let domtree = Domtree::new(code.bid_init, &cfg, &reverse_cfg);

        // Phase 3: Compute join points (phi locations)
        let (joins, joins_set) = self.compute_join_points(&stores, &inpromotable, &domtree);
        let mut join_table = JoinTable::new(&domtree, &joins_set);

        // Phase 4: Build SSA form - rename variables and determine phi nodes
        let (replaces, end_values, phinode_needed) =
            self.rename_variables(code, &inpromotable, &joins_set, &mut join_table, &domtree);

        // Phase 5: Compute values at end of blocks
        let block_end_values =
            self.compute_block_end_values(code, &inpromotable, &end_values, &replaces);

        // Phase 6: Build phi nodes
        let phinodes = self.build_phi_nodes(
            code,
            &joins_set,
            &phinode_needed,
            &end_values,
            &reverse_cfg,
            &mut join_table,
        );

        // Phase 7: Insert phi nodes and create indexes
        let phinode_indexes = self.insert_phi_nodes(code, &phinodes);

        // Phase 8: Resolve and apply replacements
        let final_block_end_values =
            self.resolve_final_values(&block_end_values, &replaces, &phinode_indexes, code);

        self.apply_replacements(code, &replaces, &phinode_indexes);

        // Phase 9: Wire up phi node arguments
        self.wire_phi_arguments(
            code,
            &phinodes,
            &phinode_indexes,
            &final_block_end_values,
            &domtree,
        );

        // Phase 10: Convert promoted operations to nops
        self.eliminate_promoted_operations(code, &inpromotable);

        true
    }
}

impl Mem2regInner {
    /// Analyzes allocations to determine which cannot be promoted
    fn analyze_allocations(
        &self,
        code: &FunctionDefinition,
    ) -> (HashSet<usize>, HashMap<usize, Vec<BlockId>>) {
        let mut inpromotable = HashSet::new();
        let mut stores = HashMap::<usize, Vec<BlockId>>::new();

        // Mark non-scalar types as inpromotable
        for (aid, alloc) in code.allocations.iter().enumerate() {
            match alloc.deref() {
                Dtype::Unit { .. }
                | Dtype::Int { .. }
                | Dtype::Float { .. }
                | Dtype::Pointer { .. } => {}
                Dtype::Array { .. }
                | Dtype::Struct { .. }
                | Dtype::Function { .. }
                | Dtype::Typedef { .. } => {
                    let _unused = inpromotable.insert(aid);
                }
            }
        }

        // Find allocations whose address is taken
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

        (inpromotable, stores)
    }

    /// Computes join points where phi nodes are needed
    fn compute_join_points(
        &self,
        stores: &HashMap<usize, Vec<BlockId>>,
        inpromotable: &HashSet<usize>,
        domtree: &Domtree,
    ) -> (HashMap<usize, HashSet<BlockId>>, HashSet<(usize, BlockId)>) {
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

        (joins, joins_set)
    }

    /// Renames variables to SSA form
    fn rename_variables(
        &self,
        code: &FunctionDefinition,
        inpromotable: &HashSet<usize>,
        joins_set: &HashSet<(usize, BlockId)>,
        join_table: &mut JoinTable<'_>,
        domtree: &Domtree,
    ) -> (
        HashMap<RegisterId, OperandVar>,
        HashMap<(usize, BlockId), OperandVar>,
        HashSet<(usize, BlockId)>,
    ) {
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

                            let var = self.find_reaching_definition(
                                *aid,
                                *bid,
                                &end_values,
                                joins_set,
                                join_table,
                                domtree,
                                code.bid_init,
                                &mut phinode_needed,
                                &code.allocations,
                            );

                            let _unused = end_values.insert((*aid, *bid), var.clone());
                            let result = replaces.insert(RegisterId::temp(*bid, i), var);
                            assert_eq!(result, None);
                        }
                    }
                    _ => {}
                }
            }
        }

        (replaces, end_values, phinode_needed)
    }

    /// Finds the reaching definition for a load
    fn find_reaching_definition(
        &self,
        aid: usize,
        bid: BlockId,
        end_values: &HashMap<(usize, BlockId), OperandVar>,
        joins_set: &HashSet<(usize, BlockId)>,
        join_table: &mut JoinTable<'_>,
        domtree: &Domtree,
        bid_init: BlockId,
        phinode_needed: &mut HashSet<(usize, BlockId)>,
        allocations: &Vec<Named<Dtype>>,
    ) -> OperandVar {
        // Check if we already have a value in this block
        if let Some(var) = end_values.get(&(aid, bid)) {
            return var.clone();
        }

        let bid_join = join_table.lookup(aid, bid);

        // Check if this is a join point
        if bid_join == bid && joins_set.contains(&(aid, bid)) {
            return OperandVar::Phi((aid, bid));
        }

        // Check if we have a value at the join point
        if let Some(val) = end_values.get(&(aid, bid_join)) {
            val.clone()
        } else {
            // Search up the dominator tree
            let mut found = None;
            let mut current = bid;

            if let Some(idom) = domtree.idom(current) {
                if joins_set.contains(&(aid, idom)) {
                    found = Some(OperandVar::Phi((aid, idom)));
                }
            }

            if found.is_none() {
                while let Some(idom) = domtree.idom(current) {
                    if let Some(val) = end_values.get(&(aid, idom)) {
                        found = Some(val.clone());
                        break;
                    }
                    if joins_set.contains(&(aid, idom)) {
                        found = Some(OperandVar::Phi((aid, idom)));
                        break;
                    }
                    current = idom;
                }
            }

            found.unwrap_or_else(|| {
                if bid_join == bid_init {
                    let dtype = allocations.get(aid).unwrap().deref();
                    OperandVar::Operand(Operand::constant(Constant::undef(dtype.clone())))
                } else {
                    let _unused = phinode_needed.insert((aid, bid_join));
                    OperandVar::Phi((aid, bid_join))
                }
            })
        }
    }

    /// Computes the values at the end of each block
    fn compute_block_end_values(
        &self,
        code: &FunctionDefinition,
        inpromotable: &HashSet<usize>,
        end_values: &HashMap<(usize, BlockId), OperandVar>,
        replaces: &HashMap<RegisterId, OperandVar>,
    ) -> HashMap<(usize, BlockId), OperandVar> {
        let mut block_end_values = end_values.clone();

        for (bid, block) in &code.blocks {
            for instr in &block.instructions {
                if let Instruction::Store { ptr, value } = instr.deref() {
                    if let Some((RegisterId::Local { aid }, _)) = ptr.get_register() {
                        if !inpromotable.contains(aid) {
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

        block_end_values
    }

    /// Builds phi nodes for all join points
    fn build_phi_nodes(
        &self,
        code: &FunctionDefinition,
        joins_set: &HashSet<(usize, BlockId)>,
        phinode_needed: &HashSet<(usize, BlockId)>,
        end_values: &HashMap<(usize, BlockId), OperandVar>,
        reverse_cfg: &HashMap<BlockId, Vec<(BlockId, JumpArg)>>,
        join_table: &mut JoinTable<'_>,
    ) -> BTreeMap<(usize, BlockId), (Dtype, HashMap<BlockId, OperandVar>)> {
        let mut phinode_visited = phinode_needed.clone();
        let mut phinode_stack = phinode_visited.iter().cloned().collect::<Vec<_>>();
        let mut phinodes = BTreeMap::new();

        // Add all join points to the stack
        for (aid, bid) in joins_set {
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

        phinodes
    }

    /// Inserts phi nodes into blocks and returns their indexes
    fn insert_phi_nodes(
        &self,
        code: &mut FunctionDefinition,
        phinodes: &BTreeMap<(usize, BlockId), (Dtype, HashMap<BlockId, OperandVar>)>,
    ) -> HashMap<(usize, BlockId), usize> {
        let mut phinode_indexes = HashMap::new();

        for ((aid, bid), (dtype, _)) in phinodes {
            let name = code.allocations.get(*aid).unwrap().name();
            let block = code.blocks.get_mut(bid).unwrap();
            let index = block.phinodes.len();
            block
                .phinodes
                .push(Named::new(name.cloned(), dtype.clone()));
            let _unused = phinode_indexes.insert((*aid, *bid), index);
        }

        phinode_indexes
    }

    /// Resolves final values by applying replacements
    fn resolve_final_values(
        &self,
        block_end_values: &HashMap<(usize, BlockId), OperandVar>,
        replaces: &HashMap<RegisterId, OperandVar>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
        code: &FunctionDefinition,
    ) -> HashMap<(usize, BlockId), OperandVar> {
        let mut final_block_end_values = HashMap::new();

        for ((aid, bid), operand_var) in block_end_values {
            let updated = match operand_var {
                OperandVar::Operand(operand) => {
                    if let Some((rid, dtype)) = operand.get_register() {
                        if let Some(replacement) = replaces.get(rid) {
                            replacement.lookup(dtype, phinode_indexes)
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

        // Resolve transitive replacements
        let mut final_replaces = HashMap::new();
        for (rid, operand_var) in replaces {
            let resolved = match operand_var {
                OperandVar::Operand(operand) => {
                    if let Some((inner_rid, _)) = operand.get_register() {
                        if let Some(inner_replacement) = replaces.get(inner_rid) {
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

        // Update block values with stores in the block
        self.update_final_block_values(
            code,
            &final_block_end_values,
            &final_replaces,
            phinode_indexes,
        )
    }

    /// Updates final block values with stores that occur within blocks
    fn update_final_block_values(
        &self,
        code: &FunctionDefinition,
        final_block_end_values: &HashMap<(usize, BlockId), OperandVar>,
        replaces: &HashMap<RegisterId, OperandVar>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
    ) -> HashMap<(usize, BlockId), OperandVar> {
        let mut result = final_block_end_values.clone();

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
                        // Apply replacements to the store value
                        let resolved_value = if let Some((rid, dtype)) = value.get_register() {
                            if let Some(replacement) = replaces.get(rid) {
                                OperandVar::Operand(replacement.lookup(dtype, phinode_indexes))
                            } else {
                                OperandVar::Operand(value.clone())
                            }
                        } else {
                            OperandVar::Operand(value.clone())
                        };
                        let _unused = current_values.insert(*aid, resolved_value);
                    }
                }
            }

            // Save final values
            for (aid, operand_var) in current_values {
                let _unused = result.insert((aid, *bid), operand_var);
            }
        }

        result
    }

    /// Applies replacements to all instructions and block exits
    fn apply_replacements(
        &self,
        code: &mut FunctionDefinition,
        replaces: &HashMap<RegisterId, OperandVar>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
    ) {
        // Resolve final replacements
        let mut final_replaces = HashMap::new();
        for (rid, operand_var) in replaces {
            let resolved = match operand_var {
                OperandVar::Operand(operand) => {
                    if let Some((inner_rid, _)) = operand.get_register() {
                        if let Some(inner_replacement) = replaces.get(inner_rid) {
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

        // Apply replacements to all uses
        for block in code.blocks.values_mut() {
            // Replace in instructions
            for instr in block.instructions.iter_mut() {
                self.replace_operands_in_instruction(instr, &final_replaces, phinode_indexes);
            }

            // Replace in block exit
            self.replace_operands_in_exit(&mut block.exit, &final_replaces, phinode_indexes);
        }
    }

    /// Replaces operands in an instruction
    fn replace_operands_in_instruction(
        &self,
        instr: &mut Named<Instruction>,
        replaces: &HashMap<RegisterId, OperandVar>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
    ) {
        match instr.deref_mut() {
            Instruction::BinOp { lhs, rhs, .. } => {
                self.replace_operand(lhs, replaces, phinode_indexes);
                self.replace_operand(rhs, replaces, phinode_indexes);
            }
            Instruction::UnaryOp { operand, .. } => {
                self.replace_operand(operand, replaces, phinode_indexes);
            }
            Instruction::Store { ptr, value } => {
                self.replace_operand(ptr, replaces, phinode_indexes);
                self.replace_operand(value, replaces, phinode_indexes);
            }
            Instruction::Load { ptr } => {
                self.replace_operand(ptr, replaces, phinode_indexes);
            }
            Instruction::Call { callee, args, .. } => {
                self.replace_operand(callee, replaces, phinode_indexes);
                for arg in args {
                    self.replace_operand(arg, replaces, phinode_indexes);
                }
            }
            Instruction::TypeCast { value, .. } => {
                self.replace_operand(value, replaces, phinode_indexes);
            }
            Instruction::GetElementPtr { ptr, offset, .. } => {
                self.replace_operand(ptr, replaces, phinode_indexes);
                self.replace_operand(offset, replaces, phinode_indexes);
            }
            _ => {}
        }
    }

    /// Replaces operands in a block exit
    fn replace_operands_in_exit(
        &self,
        exit: &mut BlockExit,
        replaces: &HashMap<RegisterId, OperandVar>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
    ) {
        match exit {
            BlockExit::ConditionalJump { condition, .. } => {
                self.replace_operand(condition, replaces, phinode_indexes);
            }
            BlockExit::Switch { value, .. } => {
                self.replace_operand(value, replaces, phinode_indexes);
            }
            BlockExit::Return { value } => {
                self.replace_operand(value, replaces, phinode_indexes);
            }
            _ => {}
        }
    }

    /// Replaces a single operand
    fn replace_operand(
        &self,
        operand: &mut Operand,
        replaces: &HashMap<RegisterId, OperandVar>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
    ) {
        if let Some((rid, dtype)) = operand.get_register() {
            if let Some(operand_var) = replaces.get(rid) {
                *operand = operand_var.lookup(dtype, phinode_indexes);
            }
        }
    }

    /// Wires up phi node arguments in predecessor blocks
    fn wire_phi_arguments(
        &self,
        code: &mut FunctionDefinition,
        phinodes: &BTreeMap<(usize, BlockId), (Dtype, HashMap<BlockId, OperandVar>)>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
        final_block_end_values: &HashMap<(usize, BlockId), OperandVar>,
        domtree: &Domtree,
    ) {
        // Create allocation to phi index mapping
        let allocation_to_phi_index = self.create_allocation_phi_mapping(code);

        // Wire up arguments for each phi node
        for ((aid, bid), (_, cases)) in phinodes {
            let phi_index = *allocation_to_phi_index
                .get(&(*aid, *bid))
                .or_else(|| phinode_indexes.get(&(*aid, *bid)))
                .expect("phi node index must exist");

            for (bid_from, operand_var) in cases {
                let block_from = code.blocks.get_mut(bid_from).unwrap();
                block_from.exit.walk_jump_args(|jump_arg| {
                    if jump_arg.bid == *bid {
                        let dtype = code.allocations.get(*aid).unwrap().deref();
                        let operand = match operand_var {
                            OperandVar::Operand(op) => op.clone(),
                            OperandVar::Phi((phi_aid, phi_bid)) => {
                                if let Some(phi_index) = phinode_indexes.get(&(*phi_aid, *phi_bid))
                                {
                                    Operand::register(
                                        RegisterId::arg(*phi_bid, *phi_index),
                                        dtype.clone(),
                                    )
                                } else {
                                    Operand::constant(Constant::undef(dtype.clone()))
                                }
                            }
                        };

                        while jump_arg.args.len() <= phi_index {
                            jump_arg.args.push(Operand::constant(Constant::unit()));
                        }
                        jump_arg.args[phi_index] = operand;
                    }
                });
            }
        }

        // Ensure all jump arguments have the right number of arguments
        self.pad_jump_arguments(code);

        // Handle complex phi dependencies
        self.resolve_complex_phi_dependencies(
            code,
            phinodes,
            phinode_indexes,
            final_block_end_values,
            domtree,
        );
    }

    /// Creates mapping from allocation IDs to phi node indexes
    fn create_allocation_phi_mapping(
        &self,
        code: &FunctionDefinition,
    ) -> HashMap<(usize, BlockId), usize> {
        let mut allocation_to_phi_index = HashMap::new();

        for (bid, block) in &code.blocks {
            for (i, phi) in block.phinodes.iter().enumerate() {
                for (aid, alloc) in code.allocations.iter().enumerate() {
                    if alloc.name() == phi.name() {
                        let _unused = allocation_to_phi_index.insert((aid, *bid), i);
                        break;
                    }
                }
            }
        }

        allocation_to_phi_index
    }

    /// Ensures all jump arguments have the correct number of arguments
    fn pad_jump_arguments(&self, code: &mut FunctionDefinition) {
        let block_phinode_counts: HashMap<BlockId, usize> = code
            .blocks
            .iter()
            .map(|(bid, block)| (*bid, block.phinodes.len()))
            .collect();

        for (_, block) in code.blocks.iter_mut() {
            block.exit.walk_jump_args(|jump_arg| {
                if let Some(&expected_args) = block_phinode_counts.get(&jump_arg.bid) {
                    while jump_arg.args.len() < expected_args {
                        jump_arg.args.push(Operand::constant(Constant::unit()));
                    }
                }
            });
        }
    }

    /// Resolves complex phi dependencies that may span multiple blocks
    fn resolve_complex_phi_dependencies(
        &self,
        code: &mut FunctionDefinition,
        phinodes: &BTreeMap<(usize, BlockId), (Dtype, HashMap<BlockId, OperandVar>)>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
        final_block_end_values: &HashMap<(usize, BlockId), OperandVar>,
        domtree: &Domtree,
    ) {
        // Build phi info structures
        let block_phi_info = self.build_block_phi_info(code, phinodes);
        let block_phi_names = self.build_block_phi_names(code);

        // Resolve arguments for each predecessor
        for (bid_from, block_from) in code.blocks.iter_mut() {
            let bid_from = *bid_from;
            block_from.exit.walk_jump_args(|jump_arg| {
                let bid_to = jump_arg.bid;
                if let Some(phi_info) = block_phi_info.get(&bid_to) {
                    while jump_arg.args.len() < phi_info.len() {
                        jump_arg.args.push(Operand::constant(Constant::unit()));
                    }

                    for (i, (aid, phi_name, dtype)) in phi_info.iter().enumerate() {
                        if *aid != usize::MAX {
                            let value = self.find_phi_value(
                                *aid,
                                bid_from,
                                phi_name,
                                dtype,
                                final_block_end_values,
                                &block_phi_names,
                                phinode_indexes,
                                domtree,
                            );
                            jump_arg.args[i] = value;
                        }
                    }

                    jump_arg.args.truncate(phi_info.len());
                }
            });
        }
    }

    /// Builds block phi info for complex resolution
    fn build_block_phi_info(
        &self,
        code: &FunctionDefinition,
        phinodes: &BTreeMap<(usize, BlockId), (Dtype, HashMap<BlockId, OperandVar>)>,
    ) -> HashMap<BlockId, Vec<(usize, String, Dtype)>> {
        code.blocks
            .iter()
            .map(|(bid, _)| {
                let mut phi_allocations = Vec::new();
                for (aid, phi_bid) in phinodes.keys() {
                    if phi_bid == bid {
                        phi_allocations.push(*aid);
                    }
                }

                phi_allocations.sort();

                let phi_info = phi_allocations
                    .into_iter()
                    .map(|aid| {
                        let alloc = code.allocations.get(aid).unwrap();
                        (
                            aid,
                            alloc.name().cloned().unwrap_or_default(),
                            alloc.deref().clone(),
                        )
                    })
                    .collect();
                (*bid, phi_info)
            })
            .collect()
    }

    /// Builds block phi names mapping
    fn build_block_phi_names(
        &self,
        code: &FunctionDefinition,
    ) -> HashMap<BlockId, Vec<Option<String>>> {
        code.blocks
            .iter()
            .map(|(bid, block)| {
                (
                    *bid,
                    block.phinodes.iter().map(|p| p.name().cloned()).collect(),
                )
            })
            .collect()
    }

    /// Finds the value for a phi node argument
    fn find_phi_value(
        &self,
        aid: usize,
        mut current_bid: BlockId,
        phi_name: &str,
        dtype: &Dtype,
        final_block_end_values: &HashMap<(usize, BlockId), OperandVar>,
        block_phi_names: &HashMap<BlockId, Vec<Option<String>>>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
        domtree: &Domtree,
    ) -> Operand {
        loop {
            if let Some(operand_var) = final_block_end_values.get(&(aid, current_bid)) {
                return operand_var.lookup(dtype, phinode_indexes);
            }

            if let Some(phi_idx) = block_phi_names.get(&current_bid).and_then(|names| {
                names
                    .iter()
                    .position(|n| n.as_ref().map(|s| s.as_str()) == Some(phi_name))
            }) {
                return Operand::register(RegisterId::arg(current_bid, phi_idx), dtype.clone());
            }

            current_bid = match domtree.idom(current_bid) {
                Some(idom) => idom,
                None => break,
            };
        }

        Operand::constant(Constant::undef(dtype.clone()))
    }

    /// Converts promoted loads and stores to nops
    fn eliminate_promoted_operations(
        &self,
        code: &mut FunctionDefinition,
        inpromotable: &HashSet<usize>,
    ) {
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
    }
}
