use core::ops::{Deref, DerefMut};
use std::collections::{BTreeMap, HashMap, HashSet};

use lang_c::visit;

use crate::ir::*;
use crate::opt::opt_utils::*;
use crate::opt::*;

use core::cmp::Ordering;

type JoinPoints = HashMap<usize, HashSet<BlockId>>;
type JoinPointsSet = HashSet<(usize, BlockId)>;
type PhiNodeData = BTreeMap<(usize, BlockId), (Dtype, HashMap<BlockId, OperandVar>)>;
type RenameResults = (
    HashMap<RegisterId, OperandVar>,
    HashMap<(usize, BlockId), OperandVar>,
    HashSet<(usize, BlockId)>,
);

struct ReachingDefContext<'a> {
    end_values: &'a HashMap<(usize, BlockId), OperandVar>,
    joins_set: &'a HashSet<(usize, BlockId)>,
    domtree: &'a Domtree,
    bid_init: BlockId,
    allocations: &'a [Named<Dtype>],
}

struct PhiValueContext<'a> {
    final_block_end_values: &'a HashMap<(usize, BlockId), OperandVar>,
    block_phi_names: &'a HashMap<BlockId, Vec<Option<String>>>,
    phinode_indexes: &'a HashMap<(usize, BlockId), usize>,
    domtree: &'a Domtree,
}

macro_rules! some_or {
    ($expr:expr, $alt:expr) => {
        match $expr {
            Some(val) => val,
            None => $alt,
        }
    };
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

fn mark_address_taken(address_taken: &mut HashSet<usize>, operand: &Operand) {
    if let Some((RegisterId::Local { aid }, _)) = operand.get_register() {
        let _unused = address_taken.insert(*aid);
    }

    if let Operand::Register {
        rid: RegisterId::Local { aid },
        ..
    } = operand
    {
        let _unused = address_taken.insert(*aid);
    }
}

impl Optimize<FunctionDefinition> for Mem2regInner {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let (inpromotable, address_taken, stores) = self.analyze_allocations(code);

        let mut changed =
            self.forward_inpromotable_store_loads(code, &inpromotable, &address_taken);

        if (0..code.allocations.len()).all(|i| inpromotable.contains(&i)) {
            return changed;
        }

        let cfg = make_cfg(code);
        let reverse_cfg = reverse_cfg(&cfg);
        let domtree = Domtree::new(code.bid_init, &cfg, &reverse_cfg);

        let (joins, joins_set) = self.compute_join_points(&stores, &inpromotable, &domtree);
        let mut join_table = JoinTable::new(&domtree, &joins_set);

        let (replaces, end_values, phinode_needed) =
            self.rename_variables(code, &inpromotable, &joins_set, &mut join_table, &domtree);

        let block_end_values =
            self.compute_block_end_values(code, &inpromotable, &end_values, &replaces);

        let phinodes = self.build_phi_nodes(
            code,
            &joins_set,
            &phinode_needed,
            &end_values,
            &reverse_cfg,
            &mut join_table,
        );

        let phinode_indexes = self.insert_phi_nodes(code, &phinodes);

        let final_block_end_values =
            self.resolve_final_values(&block_end_values, &replaces, &phinode_indexes, code);

        self.apply_replacements(code, &replaces, &phinode_indexes);

        self.wire_phi_arguments(
            code,
            &phinodes,
            &phinode_indexes,
            &final_block_end_values,
            &domtree,
        );

        self.eliminate_promoted_operations(code, &inpromotable);

        true
    }
}

impl Mem2regInner {
    fn analyze_allocations(
        &self,
        code: &FunctionDefinition,
    ) -> (HashSet<usize>, HashSet<usize>, HashMap<usize, Vec<BlockId>>) {
        let mut inpromotable = HashSet::new();
        let mut address_taken = HashSet::new();
        let mut stores = HashMap::<usize, Vec<BlockId>>::new();

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

        for (bid, block) in &code.blocks {
            for instr in &block.instructions {
                match instr.deref() {
                    Instruction::Nop => {}
                    Instruction::BinOp { lhs, rhs, .. } => {
                        mark_address_taken(&mut address_taken, lhs);
                        mark_address_taken(&mut address_taken, rhs);
                    }
                    Instruction::UnaryOp { operand, .. } => {
                        mark_address_taken(&mut address_taken, operand);
                    }
                    Instruction::Store { ptr, value } => {
                        mark_address_taken(&mut address_taken, value);

                        let (rid, _) = some_or!(ptr.get_register(), continue);
                        if let RegisterId::Local { aid } = rid {
                            stores.entry(*aid).or_default().push(*bid);
                        }
                    }
                    Instruction::Load { .. } => {}
                    Instruction::Call { callee, args, .. } => {
                        mark_address_taken(&mut address_taken, callee);
                        for arg in args {
                            mark_address_taken(&mut address_taken, arg);
                        }
                    }
                    Instruction::TypeCast { value, .. } => {
                        mark_address_taken(&mut address_taken, value);
                    }
                    _ => {}
                }
            }
        }

        for aid in &address_taken {
            let _unused = inpromotable.insert(*aid);
        }

        (inpromotable, address_taken, stores)
    }

    fn compute_join_points(
        &self,
        stores: &HashMap<usize, Vec<BlockId>>,
        inpromotable: &HashSet<usize>,
        domtree: &Domtree,
    ) -> (JoinPoints, JoinPointsSet) {
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

    fn rename_variables(
        &self,
        code: &FunctionDefinition,
        inpromotable: &HashSet<usize>,
        joins_set: &HashSet<(usize, BlockId)>,
        join_table: &mut JoinTable<'_>,
        domtree: &Domtree,
    ) -> RenameResults {
        let mut replaces = HashMap::<RegisterId, OperandVar>::new();
        let mut phinode_needed = HashSet::<(usize, BlockId)>::new();
        let mut end_values = HashMap::<(usize, BlockId), OperandVar>::new();

        for (bid, block) in &code.blocks {
            let mut current_values = HashMap::<usize, OperandVar>::new();

            for (i, instr) in block.instructions.iter().enumerate() {
                match instr.deref() {
                    Instruction::Store { ptr, value } => {
                        let (rid, _) = some_or!(ptr.get_register(), continue);
                        if let RegisterId::Local { aid } = rid {
                            if inpromotable.contains(aid) {
                                continue;
                            }

                            let stored_value = if let Some((value_rid, _)) = value.get_register() {
                                if let Some(replacement) = replaces.get(value_rid) {
                                    replacement.clone()
                                } else {
                                    OperandVar::Operand(value.clone())
                                }
                            } else {
                                OperandVar::Operand(value.clone())
                            };
                            let _unused = current_values.insert(*aid, stored_value.clone());
                            let _unused = end_values.insert((*aid, *bid), stored_value);
                        }
                    }
                    Instruction::Load { ptr } => {
                        let (rid, _) = some_or!(ptr.get_register(), continue);
                        if let RegisterId::Local { aid } = rid {
                            if inpromotable.contains(aid) {
                                continue;
                            }

                            let var = if let Some(current_var) = current_values.get(aid) {
                                current_var.clone()
                            } else {
                                let ctx = ReachingDefContext {
                                    end_values: &end_values,
                                    joins_set,
                                    domtree,
                                    bid_init: code.bid_init,
                                    allocations: &code.allocations,
                                };
                                let reaching_var = self.find_reaching_definition(
                                    *aid,
                                    *bid,
                                    &ctx,
                                    join_table,
                                    &mut phinode_needed,
                                );
                                let _unused = current_values.insert(*aid, reaching_var.clone());
                                reaching_var
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

        (replaces, end_values, phinode_needed)
    }

    fn find_reaching_definition(
        &self,
        aid: usize,
        bid: BlockId,
        ctx: &ReachingDefContext<'_>,
        join_table: &mut JoinTable<'_>,
        phinode_needed: &mut HashSet<(usize, BlockId)>,
    ) -> OperandVar {
        if let Some(var) = ctx.end_values.get(&(aid, bid)) {
            return var.clone();
        }

        if ctx.joins_set.contains(&(aid, bid)) {
            return OperandVar::Phi((aid, bid));
        }

        let bid_join = join_table.lookup(aid, bid);

        if bid_join == bid && ctx.joins_set.contains(&(aid, bid)) {
            return OperandVar::Phi((aid, bid));
        }

        if let Some(val) = ctx.end_values.get(&(aid, bid_join)) {
            val.clone()
        } else {
            let mut found = None;
            let mut current = bid;

            if let Some(idom) = ctx.domtree.idom(current) {
                if ctx.joins_set.contains(&(aid, idom)) {
                    found = Some(OperandVar::Phi((aid, idom)));
                }
            }

            if found.is_none() {
                while let Some(idom) = ctx.domtree.idom(current) {
                    if let Some(val) = ctx.end_values.get(&(aid, idom)) {
                        found = Some(val.clone());
                        break;
                    }
                    if ctx.joins_set.contains(&(aid, idom)) {
                        found = Some(OperandVar::Phi((aid, idom)));
                        break;
                    }
                    current = idom;
                }
            }

            found.unwrap_or_else(|| {
                if bid_join == ctx.bid_init {
                    let dtype = ctx.allocations.get(aid).unwrap().deref();
                    OperandVar::Operand(Operand::constant(Constant::undef(dtype.clone())))
                } else {
                    let _unused = phinode_needed.insert((aid, bid_join));
                    OperandVar::Phi((aid, bid_join))
                }
            })
        }
    }

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

    fn build_phi_nodes(
        &self,
        code: &FunctionDefinition,
        joins_set: &HashSet<(usize, BlockId)>,
        phinode_needed: &HashSet<(usize, BlockId)>,
        end_values: &HashMap<(usize, BlockId), OperandVar>,
        reverse_cfg: &HashMap<BlockId, Vec<(BlockId, JumpArg)>>,
        join_table: &mut JoinTable<'_>,
    ) -> PhiNodeData {
        let mut phinode_visited = phinode_needed.clone();
        let mut phinode_stack = phinode_visited.iter().cloned().collect::<Vec<_>>();
        let mut phinodes = BTreeMap::new();

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

    fn insert_phi_nodes(
        &self,
        code: &mut FunctionDefinition,
        phinodes: &PhiNodeData,
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

        self.update_final_block_values(
            code,
            &final_block_end_values,
            &final_replaces,
            phinode_indexes,
        )
    }

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

            for aid in 0..code.allocations.len() {
                if let Some(operand_var) = final_block_end_values.get(&(aid, *bid)) {
                    let _unused = current_values.insert(aid, operand_var.clone());
                }
            }

            for instr in &block.instructions {
                if let Instruction::Store { ptr, value } = instr.deref() {
                    if let Some((RegisterId::Local { aid }, _)) = ptr.get_register() {
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

            for (aid, operand_var) in current_values {
                let _unused = result.insert((aid, *bid), operand_var);
            }
        }

        result
    }

    fn apply_replacements(
        &self,
        code: &mut FunctionDefinition,
        replaces: &HashMap<RegisterId, OperandVar>,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
    ) {
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

        for block in code.blocks.values_mut() {
            for instr in block.instructions.iter_mut() {
                self.replace_operands_in_instruction(instr, &final_replaces, phinode_indexes);
            }

            self.replace_operands_in_exit(&mut block.exit, &final_replaces, phinode_indexes);
        }
    }

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

    fn wire_phi_arguments(
        &self,
        code: &mut FunctionDefinition,
        phinodes: &PhiNodeData,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
        final_block_end_values: &HashMap<(usize, BlockId), OperandVar>,
        domtree: &Domtree,
    ) {
        let allocation_to_phi_index = self.create_allocation_phi_mapping(code);

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

        self.pad_jump_arguments(code);

        self.resolve_complex_phi_dependencies(
            code,
            phinodes,
            phinode_indexes,
            final_block_end_values,
            domtree,
        );
    }

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

    fn resolve_complex_phi_dependencies(
        &self,
        code: &mut FunctionDefinition,
        phinodes: &PhiNodeData,
        phinode_indexes: &HashMap<(usize, BlockId), usize>,
        final_block_end_values: &HashMap<(usize, BlockId), OperandVar>,
        domtree: &Domtree,
    ) {
        let block_phi_info = self.build_block_phi_info(code, phinodes);
        let block_phi_names = self.build_block_phi_names(code);

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
                            let ctx = PhiValueContext {
                                final_block_end_values,
                                block_phi_names: &block_phi_names,
                                phinode_indexes,
                                domtree,
                            };
                            let value = self.find_phi_value(*aid, bid_from, phi_name, dtype, &ctx);
                            jump_arg.args[i] = value;
                        }
                    }

                    jump_arg.args.truncate(phi_info.len());
                }
            });
        }
    }

    fn build_block_phi_info(
        &self,
        code: &FunctionDefinition,
        phinodes: &PhiNodeData,
    ) -> HashMap<BlockId, Vec<(usize, String, Dtype)>> {
        code.blocks
            .keys()
            .map(|bid| {
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

    fn find_phi_value(
        &self,
        aid: usize,
        mut current_bid: BlockId,
        phi_name: &str,
        dtype: &Dtype,
        ctx: &PhiValueContext<'_>,
    ) -> Operand {
        loop {
            if let Some(operand_var) = ctx.final_block_end_values.get(&(aid, current_bid)) {
                return operand_var.lookup(dtype, ctx.phinode_indexes);
            }

            if let Some(phi_idx) = ctx.block_phi_names.get(&current_bid).and_then(|names| {
                names
                    .iter()
                    .position(|n| n.as_ref().map(|s| s.as_str()) == Some(phi_name))
            }) {
                return Operand::register(RegisterId::arg(current_bid, phi_idx), dtype.clone());
            }

            current_bid = match ctx.domtree.idom(current_bid) {
                Some(idom) => idom,
                None => break,
            };
        }

        Operand::constant(Constant::undef(dtype.clone()))
    }

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

    fn forward_inpromotable_store_loads(
        &self,
        code: &mut FunctionDefinition,
        inpromotable: &HashSet<usize>,
        address_taken: &HashSet<usize>,
    ) -> bool {
        let mut changed = false;
        let mut global_replacements: HashMap<RegisterId, Operand> = HashMap::new();

        for (bid, block) in code.blocks.iter_mut() {
            let mut last_stored: HashMap<usize, (usize, Operand)> = HashMap::new();
            let mut possibly_modified = HashSet::new();

            for (i, instr) in block.instructions.iter().enumerate() {
                match instr.deref() {
                    Instruction::Store { ptr, value } => {
                        if let Some((RegisterId::Local { aid }, _)) = ptr.get_register() {
                            if inpromotable.contains(aid) && !address_taken.contains(aid) {
                                let _unused = last_stored.insert(*aid, (i, value.clone()));
                            }
                        } else {
                            for aid in 0..code.allocations.len() {
                                if inpromotable.contains(&aid) && !address_taken.contains(&aid) {
                                    for j in 0..i {
                                        if let Some(instr_j) = block.instructions.get(j) {
                                            if let Instruction::GetElementPtr {
                                                ptr: gep_ptr, ..
                                            } = instr_j.deref()
                                            {
                                                if let Some((
                                                    RegisterId::Local { aid: gep_aid },
                                                    _,
                                                )) = gep_ptr.get_register()
                                                {
                                                    if *gep_aid == aid {
                                                        let _unused = possibly_modified.insert(aid);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Instruction::Load { ptr } => {
                        if let Some((RegisterId::Local { aid }, _)) = ptr.get_register() {
                            if inpromotable.contains(aid) && !address_taken.contains(aid) {
                                if let Some((store_idx, _)) = last_stored.get(aid) {}
                            }
                        }
                    }
                    Instruction::GetElementPtr { ptr, .. } => {
                        if let Some((RegisterId::Local { aid }, _)) = ptr.get_register() {
                            if inpromotable.contains(aid) && !address_taken.contains(aid) {
                                let _unused = possibly_modified.insert(*aid);
                            }
                        }
                    }
                    _ => {}
                }
            }

            let mut nop_indices = HashSet::new();

            for (i, instr) in block.instructions.iter().enumerate() {
                if let Instruction::Load { ptr } = instr.deref() {
                    if let Some((RegisterId::Local { aid }, _)) = ptr.get_register() {
                        if inpromotable.contains(aid)
                            && !address_taken.contains(aid)
                            && !possibly_modified.contains(aid)
                        {
                            if let Some((_, stored_value)) = last_stored.get(aid) {
                                let _unused = global_replacements
                                    .insert(RegisterId::temp(*bid, i), stored_value.clone());
                                let _unused = nop_indices.insert(i);
                                changed = true;
                            }
                        }
                    }
                }
            }

            for (i, instr) in block.instructions.iter_mut().enumerate() {
                if nop_indices.contains(&i) {
                    *instr.deref_mut() = Instruction::Nop;
                }
            }

            let mut store_to_nop = HashSet::new();
            for (aid, (store_idx, _)) in &last_stored {
                if possibly_modified.contains(aid) {
                    continue;
                }

                let mut only_forwarded = true;
                for (i, instr) in block.instructions.iter().enumerate() {
                    if i <= *store_idx {
                        continue;
                    }
                    if let Instruction::Load { ptr } = instr.deref() {
                        if let Some((RegisterId::Local { aid: load_aid }, _)) = ptr.get_register() {
                            if load_aid == aid && !nop_indices.contains(&i) {
                                only_forwarded = false;
                                break;
                            }
                        }
                    }
                }

                if only_forwarded {
                    let _unused = store_to_nop.insert(*store_idx);
                }
            }

            for (i, instr) in block.instructions.iter_mut().enumerate() {
                if store_to_nop.contains(&i) {
                    *instr.deref_mut() = Instruction::Nop;
                    changed = true;
                }
            }
        }

        for (_, block) in code.blocks.iter_mut() {
            for instr in block.instructions.iter_mut() {
                let _changed = instr.walk(|operand| {
                    if let Some((rid, _)) = operand.get_register() {
                        if let Some(replacement) = global_replacements.get(rid) {
                            *operand = replacement.clone();
                            return true;
                        }
                    }
                    false
                });
            }

            let _changed = block.exit.walk(|operand| {
                if let Some((rid, _)) = operand.get_register() {
                    if let Some(replacement) = global_replacements.get(rid) {
                        *operand = replacement.clone();
                        return true;
                    }
                }
                false
            });
        }

        changed
    }
}
