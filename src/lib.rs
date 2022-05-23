use std::{fmt::Display, str::FromStr};

pub use std::f64::consts::{E, PI};

#[derive(Debug, Clone)]
pub enum Function {
    Lit(f64),
    BinaryOp(Box<Function>, BinaryOperator, Box<Function>),
    UnaryOp(UnaryOperator, Box<Function>),
    Builtin(BuiltinFunction),
    Composition(Box<Function>, Box<Function>),
    Variable,
}

#[derive(Debug)]
pub enum FunctionError {
    DivideByZero,
    ZeroToZero,
}

impl Function {
    pub fn eval(&self, input: f64) -> Result<f64, FunctionError> {
        match self {
            Self::Lit(literal) => Ok(*literal),
            Self::BinaryOp(first, op, second) => {
                let first_value = first.eval(input)?;
                let second_value = second.eval(input)?;
                op.eval(first_value, second_value)
            }
            Self::UnaryOp(op, operand) => {
                let operand_value = operand.eval(input)?;
                op.eval(operand_value)
            }
            Self::Builtin(builtin) => builtin.eval(input),
            Self::Composition(outer, inner) => {
                let inner_value = inner.eval(input)?;
                outer.eval(inner_value)
            }
            Self::Variable => Ok(input),
        }
    }

    fn display_with_variable(&self, variable: &str) -> String {
        match self {
            Self::Lit(literal) => format!("{}", literal),
            Self::BinaryOp(first, op, second) => {
                let first_value = first.display_with_variable(variable);
                let second_value = second.display_with_variable(variable);
                format!("({} {} {})", first_value, op, second_value)
            }
            Self::UnaryOp(op, operand) => {
                let operand_value = operand.display_with_variable(variable);
                format!("{}{}", op, operand_value)
            }
            Self::Builtin(builtin) => format!("{}({})", builtin, variable),
            Self::Composition(outer, inner) => {
                let inner_str = inner.display_with_variable(variable);
                outer.display_with_variable(&inner_str)
            }
            Self::Variable => variable.to_string(),
        }
    }

    pub fn compose(&self, inner: &Self) -> Self {
        Self::Composition(Box::new(self.clone()), Box::new(inner.clone()))
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_with_variable("x"))
    }
}

mod string_parse {
    #[derive(Debug)]
    pub enum FunctionToken {
        RParen,
        LParen,
        Literal(f64),
        Plus,
        Minus,
        Times,
        Div,
        Pow,
        Builtin(crate::BuiltinFunction),
    }

    pub fn parse_str(s: &str) -> Result<crate::Function, FunctionParseError> {
        let tokens = tokenize_str(s);
        todo!();
    }

    fn tokenize_str(s: &str) -> Result<Vec<FunctionToken>, FunctionParseError> {
        let mut tokens = Vec::new();
        Ok(tokens)
    }

    #[derive(Debug)]
    pub enum FunctionParseError {}
}

impl FromStr for Function {
    type Err = string_parse::FunctionParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        string_parse::parse_str(s)
    }
}

#[derive(Debug, Clone)]
pub enum BinaryOperator {
    Plus,
    Minus,
    Times,
    Div,
    Pow,
}

impl BinaryOperator {
    fn eval(&self, first: f64, second: f64) -> Result<f64, FunctionError> {
        match self {
            BinaryOperator::Plus => Ok(first + second),
            BinaryOperator::Minus => Ok(first - second),
            BinaryOperator::Times => Ok(first * second),
            BinaryOperator::Div => {
                if second == 0.0 {
                    Err(FunctionError::DivideByZero)
                } else {
                    Ok(first / second)
                }
            }
            BinaryOperator::Pow => {
                if first == 0.0 && second == 0.0 {
                    Err(FunctionError::ZeroToZero)
                } else {
                    Ok(f64::powf(first, second))
                }
            }
        }
    }
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperator::Plus => write!(f, "+"),
            BinaryOperator::Minus => write!(f, "-"),
            BinaryOperator::Times => write!(f, "*"),
            BinaryOperator::Div => write!(f, "/"),
            BinaryOperator::Pow => write!(f, "^"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Negate,
}

impl UnaryOperator {
    fn eval(&self, input: f64) -> Result<f64, FunctionError> {
        match self {
            UnaryOperator::Negate => Ok(-input),
        }
    }
}

impl Display for UnaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOperator::Negate => write!(f, "-"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BuiltinFunction {
    Sin,
}

impl BuiltinFunction {
    fn eval(&self, input: f64) -> Result<f64, FunctionError> {
        match self {
            BuiltinFunction::Sin => Ok(f64::sin(input)),
        }
    }
}

impl Display for BuiltinFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuiltinFunction::Sin => write!(f, "sin"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn simple_function() {
        // 12 + x
        let func = Function::BinaryOp(
            Box::new(Function::Lit(12.0)),
            BinaryOperator::Plus,
            Box::new(Function::Variable),
        );
        println!("{}", func);
        assert_eq!(func.eval(8.0).unwrap(), 20.0);
    }

    #[test]
    fn complex_function() {
        // 5*(x + 1)^2 - 10*(x + 1) + 3
        let func = Function::Composition(
            Box::new(Function::BinaryOp(
                Box::new(Function::BinaryOp(
                    Box::new(Function::Lit(5.0)),
                    BinaryOperator::Times,
                    Box::new(Function::BinaryOp(
                        Box::new(Function::Variable),
                        BinaryOperator::Pow,
                        Box::new(Function::Lit(2.0)),
                    )),
                )),
                BinaryOperator::Plus,
                Box::new(Function::BinaryOp(
                    Box::new(Function::UnaryOp(
                        UnaryOperator::Negate,
                        Box::new(Function::BinaryOp(
                            Box::new(Function::Lit(10.0)),
                            BinaryOperator::Times,
                            Box::new(Function::Variable),
                        )),
                    )),
                    BinaryOperator::Plus,
                    Box::new(Function::Lit(3.0)),
                )),
            )),
            Box::new(Function::BinaryOp(
                Box::new(Function::Variable),
                BinaryOperator::Plus,
                Box::new(Function::Lit(1.0)),
            )),
        );
        println!("{}", func);
        let result = 5.0 * (3.5 + 1.0) * (3.5 + 1.0) - 10.0 * (3.5 + 1.0) + 3.0;
        assert_eq!(func.eval(3.5).unwrap(), result);
    }

    #[test]
    fn use_sin() {
        // sin(x)
        let func = Function::Composition(
            Box::new(Function::Builtin(BuiltinFunction::Sin)),
            Box::new(Function::Variable),
        );
        println!("{}", func);
        let result = f64::sin(2.5 * PI);
        assert_eq!(func.eval(2.5 * PI).unwrap(), result);
    }

    #[test]
    fn composition() {
        // sin(2x)
        let inner = Function::Composition(
            Box::new(Function::Builtin(BuiltinFunction::Sin)),
            Box::new(Function::BinaryOp(
                Box::new(Function::Lit(2.0)),
                BinaryOperator::Times,
                Box::new(Function::Variable),
            )),
        );
        // x^2
        let outer = Function::BinaryOp(
            Box::new(Function::Variable),
            BinaryOperator::Pow,
            Box::new(Function::Lit(2.0)),
        );
        // sin^2(2x)
        let func = outer.compose(&inner);
        println!("{}", func);
        let result = f64::sin(2.0 * 3.0) * f64::sin(2.0 * 3.0);
        assert_eq!(func.eval(3.0).unwrap(), result);
    }

    #[test]
    fn parse() {
        let func: Function = "".parse().unwrap();
        println!("{}", func);
    }
}
