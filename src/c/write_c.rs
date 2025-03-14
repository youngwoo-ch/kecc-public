use std::fmt::format;
use std::io::{Result, Write};
use std::str::FromStr;

use lang_c::span::Node;
use lang_c::{ast::*, print};

use crate::write_base::*;

impl<T: WriteLine> WriteLine for Node<T> {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        self.node.write_line(indent, write)
    }
}

impl<T: WriteString> WriteString for Node<T> {
    fn write_string(&self) -> String {
        self.node.write_string()
    }
}

impl WriteLine for TranslationUnit {
    /// VERY BIG HINT: You should start by understanding the [`writeln!`](https://doc.rust-lang.org/std/macro.writeln.html) macro.
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        for ext_decl in &self.0 {
            ext_decl.write_line(indent, write)?;
            writeln!(write)?;
        }
        Ok(())
    }
}

impl WriteLine for ExternalDeclaration {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        match self {
            Self::Declaration(decl) => decl.write_line(indent, write),
            Self::FunctionDefinition(fdef) => fdef.write_line(indent, write),
            _ => panic!("WriteLine::ExternalDeclaration"),
        }
    }
}

impl WriteLine for Declaration {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        write_indent(indent, write)?;
        writeln!(write, "{};", self.write_string())?;
        Ok(())
    }
}

impl WriteLine for InitDeclarator {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        write_indent(indent, write)?;
        writeln!(write, "{};", self.write_string())?;
        Ok(())
    }
}

impl WriteLine for FunctionDefinition {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        write_indent(indent, write)?;
        writeln!(
            write,
            "{}",
            (self.specifiers.clone(), self.declarator.clone()).write_string()
        )?;
        self.statement.write_line(indent, write)?;
        Ok(())
    }
}

impl WriteLine for Statement {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        match self {
            Self::Labeled(stmt) => {
                write_indent(indent, write)?;
                writeln!(write, "{}", stmt.node.label.write_string())?;
                stmt.node.statement.write_line(indent, write)?;
                Ok(())
            }
            Self::Compound(items) => {
                write_indent(indent, write)?;
                writeln!(write, "{{")?;
                for item in items.iter() {
                    item.write_line(indent + 1, write)?;
                }
                write_indent(indent, write)?;
                writeln!(write, "}}")?;
                Ok(())
            }
            Self::Expression(expr) => {
                write_indent(indent, write)?;
                writeln!(write, "{};", expr.as_ref().write_string())?;
                Ok(())
            }
            Self::If(stmt) => {
                write_indent(indent, write)?;
                writeln!(
                    write,
                    "if ({})",
                    stmt.node.condition.as_ref().write_string()
                )?;
                stmt.node
                    .then_statement
                    .as_ref()
                    .write_line(indent, write)?;
                if let Some(else_stmt) = &stmt.node.else_statement {
                    write_indent(indent, write)?;
                    writeln!(write, "else")?;
                    else_stmt.as_ref().write_line(indent, write)?;
                }
                Ok(())
            }
            Self::Switch(stmt) => {
                write_indent(indent, write)?;
                writeln!(
                    write,
                    "switch ({})",
                    stmt.node.expression.as_ref().write_string()
                )?;
                stmt.node.statement.as_ref().write_line(indent, write)?;
                Ok(())
            }
            Self::While(stmt) => {
                write_indent(indent, write)?;
                writeln!(
                    write,
                    "while ({})",
                    stmt.node.expression.as_ref().write_string()
                )?;
                stmt.node.statement.as_ref().write_line(indent, write)?;
                Ok(())
            }
            Self::DoWhile(stmt) => {
                write_indent(indent, write)?;
                writeln!(write, "do")?;
                stmt.node.statement.as_ref().write_line(indent, write)?;
                writeln!(
                    write,
                    "while ({});",
                    stmt.node.expression.as_ref().write_string()
                )?;
                Ok(())
            }
            Self::For(stmt) => {
                let initializer = stmt.node.initializer.write_string();
                let condition = stmt
                    .node
                    .condition
                    .as_ref()
                    .map_or("".to_string(), |c| c.write_string());
                let step = stmt
                    .node
                    .step
                    .as_ref()
                    .map_or("".to_string(), |s| s.write_string());
                write_indent(indent, write)?;
                writeln!(write, "for ({}; {}; {})", initializer, condition, step)?;
                stmt.node.statement.as_ref().write_line(indent, write)?;
                Ok(())
            }
            Self::Continue => {
                write_indent(indent, write)?;
                writeln!(write, "continue;")?;
                Ok(())
            }
            Self::Break => {
                write_indent(indent, write)?;
                writeln!(write, "break;")?;
                Ok(())
            }
            Self::Return(expr) => {
                write_indent(indent, write)?;
                writeln!(write, "return {};", expr.as_ref().write_string())?;
                Ok(())
            }
            _ => panic!("WriteLine::Statement"),
        }
    }
}

impl WriteLine for BlockItem {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        match self {
            Self::Declaration(decl) => decl.write_line(indent, write),
            Self::Statement(stmt) => stmt.write_line(indent, write),
            _ => panic!("WriteLine::BlockItem"),
        }
    }
}

impl WriteString for (Vec<Node<DeclarationSpecifier>>, Node<Declarator>) {
    fn write_string(&self) -> String {
        format!(
            "{} {}",
            self.0
                .iter()
                .map(WriteString::write_string)
                .collect::<Vec<_>>()
                .join(" "),
            self.1.write_string()
        )
    }
}

impl WriteString for Declaration {
    fn write_string(&self) -> String {
        format!(
            "{} {}",
            self.specifiers
                .iter()
                .map(WriteString::write_string)
                .collect::<Vec<_>>()
                .join(" "),
            self.declarators
                .iter()
                .map(WriteString::write_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl WriteString for DeclarationSpecifier {
    fn write_string(&self) -> String {
        match self {
            Self::StorageClass(storage_class) => storage_class.write_string(),
            Self::TypeSpecifier(type_specifier) => type_specifier.write_string(),
            Self::TypeQualifier(type_qualifier) => type_qualifier.write_string(),
            _ => panic!("WriteString::DeclarationSpecifier"),
        }
    }
}

impl WriteString for Label {
    fn write_string(&self) -> String {
        match self {
            Label::Case(expr) => format!("case {}:", expr.as_ref().write_string()),
            Label::Default => "default:".to_string(),
            _ => panic!("WriteString::Label"),
        }
    }
}

impl WriteString for StorageClassSpecifier {
    fn write_string(&self) -> String {
        match self {
            Self::Typedef => String::from("typedef"),
            _ => panic!("WriteString::StorageClassSpecifier"),
        }
    }
}

impl WriteString for TypeSpecifier {
    fn write_string(&self) -> String {
        match self {
            Self::Void => String::from("void"),
            Self::Char => String::from("char"),
            Self::Short => String::from("short"),
            Self::Int => String::from("int"),
            Self::Long => String::from("long"),
            Self::Float => String::from("float"),
            Self::Double => String::from("double"),
            Self::Signed => String::from("signed"),
            Self::Unsigned => String::from("unsigned"),
            Self::Bool => String::from("_Bool"),
            Self::Struct(struct_type) => struct_type.write_string(),
            Self::TypedefName(type_name) => type_name.write_string(),
            _ => panic!("WriteString::TypeSpecifier"),
        }
    }
}

impl WriteString for TypeQualifier {
    fn write_string(&self) -> String {
        match self {
            Self::Const => String::from("const"),
            _ => panic!("WriteString::TypeQualifier"),
        }
    }
}

impl WriteString for TypeName {
    fn write_string(&self) -> String {
        let specifiers = self
            .specifiers
            .iter()
            .map(|s| s.write_string())
            .collect::<Vec<_>>()
            .join(" ");
        let declarator = self
            .declarator
            .as_ref()
            .map(|d| d.write_string())
            .unwrap_or_default();
        format!("{} {}", specifiers, declarator).trim().to_string()
    }
}

impl WriteString for SpecifierQualifier {
    fn write_string(&self) -> String {
        match self {
            SpecifierQualifier::TypeSpecifier(spec) => spec.write_string(),
            SpecifierQualifier::TypeQualifier(qual) => qual.write_string(),
            SpecifierQualifier::Extension(exts) => panic!("WriteString::SpecifierQualifier"),
        }
    }
}

impl WriteString for InitDeclarator {
    fn write_string(&self) -> String {
        let declarator = self.declarator.write_string();
        if let Some(init) = &self.initializer {
            format!("{} = {}", declarator, init.write_string())
        } else {
            declarator.to_string()
        }
    }
}

impl WriteString for Initializer {
    fn write_string(&self) -> String {
        match self {
            Self::Expression(expr) => expr.write_string(),
            Self::List(items) => {
                let list = items
                    .iter()
                    .map(|item| item.write_string())
                    .collect::<Vec<_>>()
                    .join(", ");

                format!("{{ {} }}", list)
            }
        }
    }
}

impl WriteString for InitializerListItem {
    fn write_string(&self) -> String {
        self.initializer.as_ref().write_string()
    }
}

impl WriteString for StructType {
    fn write_string(&self) -> String {
        let kind = match self.kind.node {
            StructKind::Struct => "struct",
            StructKind::Union => "union",
        };

        let identifier = self
            .identifier
            .as_ref()
            .map(|id| format!(" {}", id.write_string()))
            .unwrap_or_default();

        let declarations = self
            .declarations
            .as_ref()
            .map(|decls| {
                let body = decls
                    .iter()
                    .map(|d| d.write_string())
                    .collect::<Vec<_>>()
                    .join("\n")
                    .lines()
                    .map(|line| format!("  {}", line))
                    .collect::<Vec<_>>()
                    .join("\n");

                format!(" {{\n{}\n}}", body)
            })
            .unwrap_or_default();

        format!("{}{}{}", kind, identifier, declarations)
    }
}

impl WriteString for StructDeclaration {
    fn write_string(&self) -> String {
        match self {
            StructDeclaration::Field(field) => format!("{};", field.write_string()),
            StructDeclaration::StaticAssert(assert) => panic!("WriteString::StructDeclaration"),
        }
    }
}

impl WriteString for StructField {
    fn write_string(&self) -> String {
        let specifiers = self
            .specifiers
            .iter()
            .map(|s| s.write_string())
            .collect::<Vec<_>>()
            .join(" ");
        let declarators = self
            .declarators
            .iter()
            .map(|d| d.write_string())
            .collect::<Vec<_>>()
            .join(", ");

        if declarators.is_empty() {
            specifiers
        } else {
            format!("{} {}", specifiers, declarators)
        }
    }
}

impl WriteString for StructDeclarator {
    fn write_string(&self) -> String {
        let declarator = self
            .declarator
            .as_ref()
            .map(|d| d.write_string())
            .unwrap_or_default();
        declarator
    }
}

impl WriteString for Expression {
    fn write_string(&self) -> String {
        match self {
            Self::Identifier(identifier) => identifier.write_string(),
            Self::Constant(constant) => constant.write_string(),
            Self::Member(member) => member.write_string(),
            Self::Call(call) => call.write_string(),
            Self::SizeOfTy(size_of_ty) => size_of_ty.write_string(),
            Self::SizeOfVal(size_of_val) => size_of_val.write_string(),
            Self::AlignOf(align_of) => align_of.write_string(),
            Self::UnaryOperator(unary) => unary.write_string(),
            Self::Cast(cast) => cast.write_string(), // 괄호로 감싸기
            Self::BinaryOperator(binary) => binary.write_string(),
            Self::Conditional(conditional) => conditional.write_string(),
            Self::Comma(exprs) => {
                let comma_expr = exprs
                    .iter()
                    .map(WriteString::write_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", comma_expr)
            }
            _ => panic!("WriteString::Expression"),
        }
    }
}

impl WriteString for Identifier {
    fn write_string(&self) -> String {
        self.name.clone()
    }
}

impl WriteString for Constant {
    fn write_string(&self) -> String {
        match self {
            Self::Integer(integer) => integer.write_string(),
            Self::Float(float) => float.write_string(),
            Self::Character(character) => character.to_string(),
        }
    }
}

impl WriteString for Integer {
    fn write_string(&self) -> String {
        let base = match self.base {
            IntegerBase::Decimal => "".to_string(),
            IntegerBase::Octal => "0".to_string(),
            IntegerBase::Hexadecimal => "0x".to_string(),
            IntegerBase::Binary => "0b".to_string(),
        };
        let suffix = self.suffix.write_string();

        format!("{}{}{}", base, self.number, suffix)
    }
}

impl WriteString for IntegerSuffix {
    fn write_string(&self) -> String {
        let mut suffix = String::new();
        if self.unsigned {
            suffix.push('U');
        }
        suffix.push_str(&self.size.write_string());

        suffix
    }
}

impl WriteString for IntegerSize {
    fn write_string(&self) -> String {
        match self {
            IntegerSize::Int => "".to_string(),
            IntegerSize::Long => "L".to_string(),
            IntegerSize::LongLong => "LL".to_string(),
        }
    }
}

impl WriteString for Float {
    fn write_string(&self) -> String {
        let base = match self.base {
            FloatBase::Decimal => "".to_string(),
            FloatBase::Hexadecimal => "0x".to_string(),
        };
        let suffix = self.suffix.write_string();

        format!("{}{}{}", base, self.number, suffix)
    }
}

impl WriteString for FloatSuffix {
    fn write_string(&self) -> String {
        let mut suffix = self.format.write_string();
        if self.imaginary {
            suffix.push('i');
        }

        suffix
    }
}

impl WriteString for FloatFormat {
    fn write_string(&self) -> String {
        match self {
            FloatFormat::Float => "f".to_string(),
            FloatFormat::Double => "".to_string(),
            FloatFormat::LongDouble => "l".to_string(),
            FloatFormat::TS18661Format(ts_type) => panic!("WriteString::FloatFormat"),
        }
    }
}

impl WriteString for BinaryOperatorExpression {
    fn write_string(&self) -> String {
        match self.operator.node {
            BinaryOperator::Index => {
                format!(
                    "{}[{}]",
                    self.lhs.node.write_string(),
                    self.rhs.node.write_string()
                )
            }
            _ => match self.operator.node {
                BinaryOperator::Index
                | BinaryOperator::Assign
                | BinaryOperator::AssignMultiply
                | BinaryOperator::AssignDivide
                | BinaryOperator::AssignModulo
                // | BinaryOperator::AssignPlus
                // | BinaryOperator::AssignMinus
                | BinaryOperator::AssignShiftLeft
                | BinaryOperator::AssignShiftRight
                | BinaryOperator::AssignBitwiseAnd
                | BinaryOperator::AssignBitwiseOr
                | BinaryOperator::AssignBitwiseXor => {
                    format!(
                        "{} {} {}",
                        self.lhs.node.write_string(),
                        self.operator.node.write_string(),
                        self.rhs.node.write_string()
                    )
                }
                _ => {
                    format!(
                        "(({}) {} ({}))",
                        self.lhs.node.write_string(),
                        self.operator.node.write_string(),
                        self.rhs.node.write_string()
                    )
                }
            },
        }
    }
}

impl WriteString for BinaryOperator {
    fn write_string(&self) -> String {
        match self {
            Self::Index => String::from("[]"),
            Self::Multiply => String::from("*"),
            Self::Divide => String::from("/"),
            Self::Modulo => String::from("%"),
            Self::Plus => String::from("+"),
            Self::Minus => String::from("-"),
            Self::ShiftLeft => String::from("<<"),
            Self::ShiftRight => String::from(">>"),
            Self::Less => String::from("<"),
            Self::Greater => String::from(">"),
            Self::LessOrEqual => String::from("<="),
            Self::GreaterOrEqual => String::from(">="),
            Self::Equals => String::from("=="),
            Self::NotEquals => String::from("!="),
            Self::BitwiseAnd => String::from("&"),
            Self::BitwiseXor => String::from("^"),
            Self::BitwiseOr => String::from("|"),
            Self::LogicalAnd => String::from("&&"),
            Self::LogicalOr => String::from("||"),
            Self::Assign => String::from("="),
            Self::AssignMultiply => String::from("*="),
            Self::AssignDivide => String::from("/="),
            Self::AssignModulo => String::from("%="),
            Self::AssignPlus => String::from("+="),
            Self::AssignMinus => String::from("-="),
            Self::AssignShiftLeft => String::from("<<="),
            Self::AssignShiftRight => String::from(">>="),
            Self::AssignBitwiseAnd => String::from("&="),
            Self::AssignBitwiseXor => String::from("^="),
            Self::AssignBitwiseOr => String::from("|="),
        }
    }
}

impl WriteString for ForInitializer {
    fn write_string(&self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Expression(expr) => expr.as_ref().write_string(),
            Self::Declaration(decl) => decl.write_string(),
            Self::StaticAssert(_) => panic!("WriteString::ForInitializer"),
        }
    }
}

impl WriteString for Declarator {
    fn write_string(&self) -> String {
        let kind = self.kind.write_string();

        let pointer_count = self
            .derived
            .iter()
            .filter(|d| matches!(d.node, DerivedDeclarator::Pointer(_)))
            .count();

        let has_kr_function = self
            .derived
            .iter()
            .any(|d| matches!(d.node, DerivedDeclarator::KRFunction(_)));

        let derived = self
            .derived
            .iter()
            .map(WriteString::write_string)
            .collect::<Vec<_>>()
            .join("");

        if pointer_count > 0 && has_kr_function {
            let pointers = "*".repeat(pointer_count);
            format!("({}{}{})", pointers, kind, derived)
        } else if pointer_count > 0 {
            let pointers = "*".repeat(pointer_count);
            format!("{}{}{}", pointers, kind, derived)
        } else {
            format!("{}{}", kind, derived)
        }
    }
}

impl WriteString for DeclaratorKind {
    fn write_string(&self) -> String {
        match self {
            Self::Abstract => String::from(""),
            Self::Identifier(identifier) => identifier.write_string(),
            Self::Declarator(decl) => decl.write_string(),
        }
    }
}

impl WriteString for DerivedDeclarator {
    fn write_string(&self) -> String {
        match self {
            Self::Pointer(pointer_qualifiers) => String::new(),
            Self::Array(array_decl) => array_decl.write_string(),
            Self::Function(func_decl) => func_decl.write_string(),
            Self::KRFunction(_) => String::from("()"),
            _ => panic!("WriteString::DerivedDeclarator"),
        }
    }
}

impl WriteString for ArrayDeclarator {
    fn write_string(&self) -> String {
        match &self.size {
            ArraySize::VariableExpression(expr) => format!("[{}]", expr.as_ref().write_string()),
            _ => panic!("WriteString::ArrayDeclarator"),
        }
    }
}

impl WriteString for FunctionDeclarator {
    fn write_string(&self) -> String {
        format!(
            "({})",
            self.parameters
                .iter()
                .map(WriteString::write_string)
                .collect::<Vec<_>>()
                .join(", "),
        )
    }
}

impl WriteString for ParameterDeclaration {
    fn write_string(&self) -> String {
        let specifiers = self
            .specifiers
            .iter()
            .map(WriteString::write_string)
            .collect::<Vec<_>>()
            .join(" ");
        if let Some(decl) = &self.declarator {
            format!("{} {}", specifiers, decl.node.write_string())
        } else {
            specifiers.to_string()
        }
    }
}

impl WriteString for UnaryOperatorExpression {
    fn write_string(&self) -> String {
        let operand = self.operand.as_ref().write_string();
        match self.operator.node {
            UnaryOperator::PostIncrement => format!("{}++", operand),
            UnaryOperator::PostDecrement => format!("{}--", operand),
            UnaryOperator::PreIncrement => format!("++{}", operand),
            UnaryOperator::PreDecrement => format!("--{}", operand),
            UnaryOperator::Address => format!("&{}", operand),
            UnaryOperator::Indirection => format!("*{}", operand),
            UnaryOperator::Plus => format!("+{}", operand),
            UnaryOperator::Minus => format!("-{}", operand),
            UnaryOperator::Complement => format!("~({})", operand),
            UnaryOperator::Negate => format!("!({})", operand),
        }
    }
}

impl WriteString for ConditionalExpression {
    fn write_string(&self) -> String {
        let condition = self.condition.as_ref().write_string();
        let then_expr = self.then_expression.as_ref().write_string();
        let else_expr = self.else_expression.as_ref().write_string();
        format!("({} ? {} : {})", condition, then_expr, else_expr)
    }
}

impl WriteString for CastExpression {
    fn write_string(&self) -> String {
        let type_name = self.type_name.write_string();
        let expression = self.expression.as_ref().write_string();
        format!("({}) {}", type_name, expression)
    }
}

impl WriteString for AlignOf {
    fn write_string(&self) -> String {
        format!("_Alignof({})", self.0.as_ref().write_string())
    }
}

impl WriteString for MemberExpression {
    fn write_string(&self) -> String {
        let expr = self.expression.as_ref().write_string();
        let op = match self.operator.node {
            MemberOperator::Direct => ".",
            MemberOperator::Indirect => "->",
        };
        let id = self.identifier.write_string();
        format!("{}{}{}", expr, op, id)
    }
}

impl WriteString for CallExpression {
    fn write_string(&self) -> String {
        let callee = self.callee.as_ref().write_string();
        let arguments = self
            .arguments
            .iter()
            .map(|arg| arg.write_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}({})", callee, arguments)
    }
}

impl WriteString for SizeOfTy {
    fn write_string(&self) -> String {
        format!("sizeof({})", self.0.write_string())
    }
}

impl WriteString for SizeOfVal {
    fn write_string(&self) -> String {
        format!("sizeof({})", self.0.as_ref().write_string())
    }
}
