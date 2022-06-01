#![warn(missing_docs)]

//! A mathematical library for creating, operating on, and evaluating functions.
//!
//! The main data type used is [`Function`], which represents a part of or an entire function.
//!
//! # Defining and evaluating a function
//!
//! ### Manual definition
//! [`Function`]s can be defined part by part like so.
//! ```
//! # use function::{Function, BuiltinFunction, BinaryOperator, PI};
//! // sin^2(x)
//! let func = Function::BinaryOp(
//!     Box::new(Function::Builtin(BuiltinFunction::Sin, Box::new(Function::Variable))),
//!     BinaryOperator::Pow,
//!     Box::new(Function::Lit(2.0)),
//! );
//! assert!(func.eval(PI).unwrap() < 1e-15);
//! ```
//! This is very verbose way to define a function, as it is very verbose,
//! so defining functions this way is not recommended. It is only recommended
//! if you want to specifically define and optimize your functions.
//!
//! ### Automatic parsing
//! [`Function`] implements [`FromStr`] and therefore functions can be derived from strings.
//! ```
//! # use function::{Function,PI};
//! let func: Function = "sin(x)^2".parse().unwrap();
//! assert!(func.eval(PI).unwrap() < 1e-15);
//! ```
//!
//! [`FromStr`]: std::str::FromStr

use std::{fmt::Display, rc::Rc};
use std::str::FromStr;

pub use std::f64::consts::{E, PI};

/// The core data type of the crate.
///
/// A `Function` can be evaluated for a given value using [`eval`].
///
/// [`eval`]: fn@crate::Function::eval
#[derive(Debug, Clone, PartialEq)]
pub enum Function {
    /// A numeric literal. All literal values are represented as 64-bt floats.
    Lit(f64),
    /// A binary operation (ex. `f(x) + g(x)`).
    BinaryOp(Box<Function>, BinaryOperator, Box<Function>),
    /// A unary operation (ex. `-f(x)`).
    UnaryOp(UnaryOperator, Box<Function>),
    /// A builtin function such as `sin`.
    Builtin(BuiltinFunction, Box<Function>),
    /// A composition of functions (ex. `f(g(x))`).
    Composition(Box<Function>, Box<Function>),
    /// Represents the variable in a function (ex. the `x` in `x^2`).
    /// In a composition of functions, `Variable` inside the outer function represents the entire inner function.
    Variable,
    /// Used to hold a reference to an existing function, which you wish to keep access to outside this function.
    /// If you don't need to keep the reference outside of this function, I recommend moving it into this function
    /// inside a Box in one of the other variants. Using this variant adds an extra level of indirection through the Rc
    /// to using the held function, both for this and other uses of it.
    Existing(Rc<Function>),
}

/// Enumeration of all errors which can occur while evaluating a function.
#[derive(Debug, PartialEq, Eq)]
pub enum FunctionEvaluationError {
    /// Caused by division by zero.
    DivideByZero,
    /// Caused by `0^0`.
    ZeroToZero,
}

impl Function {
    /// Evaluate the function with the given value.
    ///
    /// # Errors
    ///
    /// This function will return an error if an illegal operation takes place during evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use function::{Function, BinaryOperator};
    /// // x^2 - 1
    /// let func = Function::BinaryOp(
    ///     Box::new(Function::BinaryOp(
    ///         Box::new(Function::Variable),
    ///         BinaryOperator::Pow,
    ///         Box::new(Function::Lit(2.0)),
    ///     )),
    ///     BinaryOperator::Minus,
    ///     Box::new(Function::Lit(1.0)),
    /// );
    /// assert_eq!(func.eval(2.0).unwrap(), 3.0);
    /// ```
    ///
    /// ```
    /// # use function::{Function, BinaryOperator, FunctionEvaluationError};
    /// // 1/x
    /// let func = Function::BinaryOp(
    ///     Box::new(Function::Lit(1.0)),
    ///     BinaryOperator::Div,
    ///     Box::new(Function::Variable),
    /// );
    /// assert_eq!(func.eval(0.0).unwrap_err(), FunctionEvaluationError::DivideByZero);
    pub fn eval(&self, value: f64) -> Result<f64, FunctionEvaluationError> {
        match self {
            Self::Lit(literal) => Ok(*literal),
            Self::BinaryOp(first, op, second) => {
                let first_value = first.eval(value)?;
                let second_value = second.eval(value)?;
                op.eval(first_value, second_value)
            }
            Self::UnaryOp(op, operand) => {
                let operand_value = operand.eval(value)?;
                op.eval(operand_value)
            }
            Self::Builtin(builtin, inner) => {
                let input_value = inner.eval(value)?;
                builtin.eval(input_value)
            }
            Self::Composition(outer, inner) => {
                let inner_value = inner.eval(value)?;
                outer.eval(inner_value)
            }
            Self::Variable => Ok(value),
            Self::Existing(other) => other.eval(value),
        }
    }

    fn eval_literals(self) -> Result<Self, FunctionEvaluationError> {
        match self {
            Self::BinaryOp(left, op, right) => {
                let left = left.eval_literals()?;
                let right = right.eval_literals()?;
                if let (&Self::Lit(left), &Self::Lit(right)) = (&left, &right) {
                    Ok(Self::Lit(op.eval(left, right)?))
                } else {
                    Ok(Self::BinaryOp(Box::new(left), op, Box::new(right)))
                }
            }
            Self::UnaryOp(op, operand) => {
                let operand = operand.eval_literals()?;
                if let &Self::Lit(operand) = &operand {
                    Ok(Self::Lit(op.eval(operand)?))
                } else {
                    Ok(Self::UnaryOp(op, Box::new(operand)))
                }
            }
            Self::Builtin(builtin, inside) => {
                let inside = inside.eval_literals()?;
                if let &Self::Lit(inside) = &inside {
                    Ok(Self::Lit(builtin.eval(inside)?))
                } else {
                    Ok(Self::Builtin(builtin, Box::new(inside)))
                }
            }
            Self::Composition(outer, inner) => {
                let inner = inner.eval_literals()?;
                let outer = outer.eval_literals()?;
                if let &Self::Lit(outer) = &outer {
                    Ok(Self::Lit(outer))
                } else if let &Self::Lit(inner) = &inner {
                    Ok(Self::Lit(outer.eval(inner)?))
                } else {
                    Ok(Self::Composition(Box::new(outer), Box::new(inner)))
                }
            }
            Self::Lit(_) | Self::Variable => Ok(self),
            Self::Existing(_) => Ok(self),
        }
    }

    /// Attempts to simplify a function to make eventual evaluations faster.
    /// Goes through the following steps to optimize the function:
    /// * literal evaluation
    ///     - Traverses the function and evaluates operations that can be performed at "compile time."
    ///     For example, the function `1 + 1` reduces to a literal `2`.
    ///     Currently this does not handle literals which, though mathematically able to be evaluated,
    ///     are not adjacent. `x + 1 + 1` would not reduce currently as the parser handles operators as
    ///     left-associative, grouping the first 1 with the x rather than with the second 1.
    /// ```
    /// # use function::Function;
    /// let func: Function = "1 + 1" .parse().unwrap();
    /// let reduced = func.reduce().unwrap();
    /// let expected: Function = "2".parse().unwrap();
    /// assert_eq!(reduced, expected);
    /// ```
    /// # Errors
    /// If in the process of simplifying the function, an evaluation error occurs
    /// (such as in the case of the function `1/0`)
    /// the reduction will fail and return an Err with the corresponding error.
    /// ```
    /// # use function::{Function, FunctionEvaluationError};
    /// let func: Function = "1/0".parse().unwrap();
    /// assert_eq!(func.reduce().unwrap_err(), FunctionEvaluationError::DivideByZero);
    /// ```
    pub fn reduce(&self) -> Result<Self, FunctionEvaluationError> {
        let new = self.clone();
        new.eval_literals()
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
            Self::Builtin(builtin, inner) => {
                format!("{}({})", builtin, inner.display_with_variable(variable))
            }
            Self::Composition(outer, inner) => {
                let inner_str = inner.display_with_variable(variable);
                outer.display_with_variable(&inner_str)
            }
            Self::Variable => variable.to_string(),
            Self::Existing(other) => other.display_with_variable(variable),
        }
    }

    /// Returns a composition of functions in which `self` is the outer function and `inner` is the inner function.
    ///
    /// # Examples
    /// ```
    /// # use function::{Function, BuiltinFunction, BinaryOperator};
    /// // sin(x)
    /// let inner = Function::Builtin(
    ///     BuiltinFunction::Sin,
    ///     Box::new(Function::Variable)
    /// );
    /// // x^2
    /// let outer = Function::BinaryOp(
    ///     Box::new(Function::Variable),
    ///     BinaryOperator::Pow,
    ///     Box::new(Function::Lit(2.0)),
    /// );
    /// // sin^2(x)
    /// let func = outer.compose(&inner);
    /// let result = f64::sin(3.0) * f64::sin(3.0);
    /// assert_eq!(func.eval(3.0).unwrap(), result);
    /// ```
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
    use std::iter::Peekable;

    use crate::{BinaryOperator, BuiltinFunction, Function, UnaryOperator};
    type Result<T> = std::result::Result<T, FunctionParseError>;

    #[derive(Debug)]
    struct FunctionToken {
        kind: FunctionTokenKind,
        pos: usize,
    }

    #[derive(Debug)]
    enum FunctionTokenKind {
        RParen,
        LParen,
        Literal(f64),
        Plus,
        Minus,
        Times,
        Div,
        Pow,
        Builtin(BuiltinFunction),
        Variable,
        Eof,
    }

    macro_rules! token_iter {
        () => {
            Peekable<impl Iterator<Item = FunctionToken>>
        };
    }

    fn item(tokens: &mut token_iter!()) -> Result<Function> {
        let next_token = tokens.next().expect("Ran out of tokens");
        match next_token.kind {
            FunctionTokenKind::LParen => {
                let item = add_expr(tokens)?;
                let next_token = tokens.next().expect("Ran out of tokens");
                if let FunctionTokenKind::RParen = next_token.kind {
                    Ok(item)
                } else {
                    Err(FunctionParseError {
                        cause: FunctionParseErrorCause::UnmatchedParentheses,
                        position: next_token.pos,
                    })
                }
            }
            FunctionTokenKind::Literal(lit) => Ok(Function::Lit(lit)),
            FunctionTokenKind::Builtin(builtin) => {
                if let FunctionTokenKind::LParen = tokens.next().expect("Ran out of tokens").kind {
                    let item = add_expr(tokens)?;
                    let next_token = tokens.next().expect("Ran out of tokens");
                    if let FunctionTokenKind::RParen = next_token.kind {
                        Ok(Function::Builtin(builtin, Box::new(item)))
                    } else {
                        Err(FunctionParseError {
                            cause: FunctionParseErrorCause::UnexpectedCharacter,
                            position: next_token.pos,
                        })
                    }
                } else {
                    Err(FunctionParseError {
                        cause: FunctionParseErrorCause::UnexpectedCharacter,
                        position: next_token.pos,
                    })
                }
            }
            FunctionTokenKind::Variable => Ok(Function::Variable),
            FunctionTokenKind::Minus => {
                let operand = item(tokens)?;
                Ok(Function::UnaryOp(UnaryOperator::Negate, Box::new(operand)))
            }
            FunctionTokenKind::RParen
            | FunctionTokenKind::Plus
            | FunctionTokenKind::Times
            | FunctionTokenKind::Div
            | FunctionTokenKind::Pow
            | FunctionTokenKind::Eof => Err(FunctionParseError {
                cause: FunctionParseErrorCause::UnexpectedToken,
                position: next_token.pos,
            }),
        }
    }

    fn pow_expr(tokens: &mut token_iter!()) -> Result<Function> {
        let mut pow_tree = Vec::new();
        pow_tree.push(item(tokens)?);
        while let Some(FunctionToken {
            kind: FunctionTokenKind::Pow,
            pos: _,
        }) = tokens.peek()
        {
            tokens.next().unwrap();
            pow_tree.push(item(tokens)?);
        }
        let mut result = pow_tree.pop().unwrap();
        while let Some(base) = pow_tree.pop() {
            result = Function::BinaryOp(Box::new(base), BinaryOperator::Pow, Box::new(result));
        }
        Ok(result)
    }

    fn mul_expr(tokens: &mut token_iter!()) -> Result<Function> {
        let mut lhs = pow_expr(tokens)?;
        while let Some(FunctionToken {
            kind: FunctionTokenKind::Times | FunctionTokenKind::Div,
            pos: _,
        }) = tokens.peek()
        {
            let op_token = tokens.next().unwrap();
            let rhs = pow_expr(tokens)?;
            lhs = Function::BinaryOp(
                Box::new(lhs),
                match op_token.kind {
                    FunctionTokenKind::Times => BinaryOperator::Times,
                    FunctionTokenKind::Div => BinaryOperator::Div,
                    _ => unreachable!(),
                },
                Box::new(rhs),
            )
        }
        Ok(lhs)
    }

    fn add_expr(tokens: &mut token_iter!()) -> Result<Function> {
        let mut lhs = mul_expr(tokens)?;
        while let Some(FunctionToken {
            kind: FunctionTokenKind::Plus | FunctionTokenKind::Minus,
            pos: _,
        }) = tokens.peek()
        {
            let op_token = tokens.next().unwrap();
            let rhs = mul_expr(tokens)?;
            lhs = Function::BinaryOp(
                Box::new(lhs),
                match op_token.kind {
                    FunctionTokenKind::Plus => BinaryOperator::Plus,
                    FunctionTokenKind::Minus => BinaryOperator::Minus,
                    _ => unreachable!(),
                },
                Box::new(rhs),
            );
        }
        Ok(lhs)
    }

    fn function(tokens: &mut token_iter!()) -> Result<Function> {
        let expr = add_expr(tokens)?;
        let last_token = tokens.next().expect("Ran out of tokens");
        if let FunctionTokenKind::Eof = last_token.kind {
            Ok(expr)
        } else {
            Err(FunctionParseError {
                cause: FunctionParseErrorCause::TrailingTokens,
                position: last_token.pos,
            })
        }
    }

    pub fn parse_str(s: &str) -> Result<Function> {
        let tokens = tokenize_str(s)?;
        function(&mut tokens.into_iter().peekable())
    }

    fn tokenize_str(s: &str) -> Result<Vec<FunctionToken>> {
        #[derive(Debug)]
        enum TokenState {
            Start,
            InNum,
            InBuiltin,
        }

        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut current_state = TokenState::Start;
        let mut token_start = 0;
        for (i, c) in s.chars().enumerate() {
            let mut repeat = true;
            while repeat {
                repeat = false;
                match current_state {
                    TokenState::Start => {
                        if c.is_numeric() || c == '.' {
                            current_token.push(c);
                            current_state = TokenState::InNum;
                            token_start = i;
                            continue;
                        }
                        if c == 'x' {
                            tokens.push(FunctionToken {
                                kind: FunctionTokenKind::Variable,
                                pos: i,
                            });
                            continue;
                        }
                        if c.is_alphabetic() {
                            current_token.push(c);
                            current_state = TokenState::InBuiltin;
                            token_start = i;
                            continue;
                        }
                        if c.is_whitespace() {
                            continue;
                        }
                        match c {
                            '(' => {
                                tokens.push(FunctionToken {
                                    kind: FunctionTokenKind::LParen,
                                    pos: i,
                                });
                            }
                            ')' => {
                                tokens.push(FunctionToken {
                                    kind: FunctionTokenKind::RParen,
                                    pos: i,
                                });
                            }
                            '+' => {
                                tokens.push(FunctionToken {
                                    kind: FunctionTokenKind::Plus,
                                    pos: i,
                                });
                            }
                            '-' => {
                                tokens.push(FunctionToken {
                                    kind: FunctionTokenKind::Minus,
                                    pos: i,
                                });
                            }
                            '*' => {
                                tokens.push(FunctionToken {
                                    kind: FunctionTokenKind::Times,
                                    pos: i,
                                });
                            }
                            '/' => {
                                tokens.push(FunctionToken {
                                    kind: FunctionTokenKind::Div,
                                    pos: i,
                                });
                            }
                            '^' => {
                                tokens.push(FunctionToken {
                                    kind: FunctionTokenKind::Pow,
                                    pos: i,
                                });
                            }
                            _ => {
                                return Err(FunctionParseError {
                                    cause: FunctionParseErrorCause::UnexpectedCharacter,
                                    position: i,
                                });
                            }
                        }
                    }
                    TokenState::InNum => {
                        if c.is_numeric() || c == '.' {
                            current_token.push(c);
                        } else {
                            match current_token.parse() {
                                Ok(num) => tokens.push(FunctionToken {
                                    kind: FunctionTokenKind::Literal(num),
                                    pos: token_start,
                                }),
                                Err(_) => {
                                    return Err(FunctionParseError {
                                        cause: FunctionParseErrorCause::LiteralParseFailure,
                                        position: i - 1,
                                    })
                                }
                            }
                            current_token.clear();
                            current_state = TokenState::Start;
                            repeat = true;
                        }
                    }
                    TokenState::InBuiltin => match current_token.as_str() {
                        "pi" => {
                            tokens.push(FunctionToken {
                                kind: FunctionTokenKind::Literal(crate::PI),
                                pos: token_start,
                            });
                            current_token.clear();
                            current_state = TokenState::Start;
                            repeat = true;
                        }
                        "e" => {
                            tokens.push(FunctionToken {
                                kind: FunctionTokenKind::Literal(crate::E),
                                pos: token_start,
                            });
                            current_token.clear();
                            current_state = TokenState::Start;
                            repeat = true;
                        }
                        _ => {
                            if c == '(' {
                                if let Ok(b) = current_token.parse::<BuiltinFunction>() {
                                    tokens.push(FunctionToken {
                                        kind: FunctionTokenKind::Builtin(b),
                                        pos: token_start,
                                    });
                                    current_token.clear();
                                    current_state = TokenState::Start;
                                    repeat = true;
                                } else {
                                    return Err(FunctionParseError {
                                        cause: FunctionParseErrorCause::BuiltinParseFailure,
                                        position: i,
                                    });
                                }
                            } else {
                                current_token.push(c);
                            }
                        }
                    },
                }
            }
        }
        if !current_token.is_empty() {
            if let Ok(lit) = current_token.parse() {
                tokens.push(FunctionToken {
                    kind: FunctionTokenKind::Literal(lit),
                    pos: s.len() - 1,
                })
            } else if let Ok(builtin) = current_token.parse() {
                tokens.push(FunctionToken {
                    kind: FunctionTokenKind::Builtin(builtin),
                    pos: s.len() - 1,
                })
            } else {
                return Err(FunctionParseError {
                    cause: FunctionParseErrorCause::TrailingTokens,
                    position: s.len() - current_token.len(),
                });
            }
        }
        tokens.push(FunctionToken {
            kind: FunctionTokenKind::Eof,
            pos: s.len(),
        });
        Ok(tokens)
    }

    /// Error type for function string parsing.
    #[derive(Debug)]
    pub struct FunctionParseError {
        /// Cause of the error.
        pub cause: FunctionParseErrorCause,
        /// Index in the string at which the error occurred.
        pub position: usize,
    }

    /// Enumeration of possible causes for parse errors.
    #[derive(Debug)]
    pub enum FunctionParseErrorCause {
        /// A character was parsed that is not understood by the parser.
        UnexpectedCharacter,
        /// There was an issue in parsing a numeric literal.
        LiteralParseFailure,
        /// There [built-in] function was not recognized.
        ///
        /// [built-in]: enum@crate::BuiltinFunction
        BuiltinParseFailure,
        /// The function finished parsing before the end of the string.
        TrailingTokens,
        /// The parser expected an item at the location, but couldn't parse one.
        ExpectedItem,
        /// The parser encountered a token it didn't expect at that point.
        UnexpectedToken,
        /// A left paren was not paired with a matching right paren.
        UnmatchedParentheses,
    }
}
pub use string_parse::{FunctionParseError, FunctionParseErrorCause};

impl FromStr for Function {
    type Err = string_parse::FunctionParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        string_parse::parse_str(s)
    }
}

/// Enumeration of binary operators for use in [`Function::BinaryOp`]
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    /// Addition.
    Plus,
    /// Subtraction.
    Minus,
    /// Multiplication.
    Times,
    /// Division.
    Div,
    /// Exponentiation.
    Pow,
}

impl BinaryOperator {
    fn eval(&self, first: f64, second: f64) -> Result<f64, FunctionEvaluationError> {
        match self {
            BinaryOperator::Plus => Ok(first + second),
            BinaryOperator::Minus => Ok(first - second),
            BinaryOperator::Times => Ok(first * second),
            BinaryOperator::Div => {
                if second == 0.0 {
                    Err(FunctionEvaluationError::DivideByZero)
                } else {
                    Ok(first / second)
                }
            }
            BinaryOperator::Pow => {
                if first == 0.0 && second == 0.0 {
                    Err(FunctionEvaluationError::ZeroToZero)
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

/// Enumeration of unary operators for use in [`Function::UnaryOp`]
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    /// Negation
    Negate,
}

impl UnaryOperator {
    fn eval(&self, input: f64) -> Result<f64, FunctionEvaluationError> {
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

/// Enumeration of built in functions for use in [`Function::Builtin`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuiltinFunction {
    /// [Sine] function.
    ///
    /// [Sine]: https://en.wikipedia.org/wiki/Sine_and_cosine
    Sin,
}

impl BuiltinFunction {
    fn eval(&self, input: f64) -> Result<f64, FunctionEvaluationError> {
        match self {
            BuiltinFunction::Sin => Ok(f64::sin(input)),
        }
    }
}

impl FromStr for BuiltinFunction {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sin" => Ok(BuiltinFunction::Sin),
            _ => Err(()),
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
        let func = Function::Builtin(BuiltinFunction::Sin, Box::new(Function::Variable));
        println!("{}", func);
        let result = f64::sin(2.5 * PI);
        assert_eq!(func.eval(2.5 * PI).unwrap(), result);
    }

    #[test]
    fn composition() {
        // sin(2x)
        let inner = Function::Builtin(
            BuiltinFunction::Sin,
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
        let func: Function = "1 + x".parse().unwrap();
        println!("{}", func);
        let expected = Function::BinaryOp(
            Box::new(Function::Lit(1.0)),
            BinaryOperator::Plus,
            Box::new(Function::Variable),
        );
        assert_eq!(func, expected);
    }

    #[test]
    fn parse_complex() {
        let func: Function = "5*(x + 1)^2".parse().unwrap();
        println!("{}", func);
    }

    #[test]
    fn parse_sin() {
        let func: Function = "sin(pi*x)".parse().unwrap();
        assert_eq!(func.eval(1.0).unwrap(), f64::sin(PI));
    }

    #[test]
    fn literal_reduction() {
        let func: Function = "(sin(pi) + 2^2) + x".parse().unwrap();
        let reduced_func = func.reduce().unwrap();
        println!("{}\n{}", func, reduced_func);
        let expected: Function = "4+x".parse().unwrap();
        assert_eq!(reduced_func, expected);
    }

    #[test]
    fn sum_three() {
        let func: Function = "1 + 1 + 1".parse().unwrap();
        assert_eq!(func.eval(0.0).unwrap(), 3.0);
    }

    #[test]
    fn sum_many() {
        let func: Function = "-1 + 1 - 1 + 2*3^2*1".parse().unwrap();
        assert_eq!(func.eval(0.0).unwrap(), 17.0);
    }

    #[test]
    fn pow_right_associative() {
        // 3 ^ (2^3)
        // 3^8
        // 6561
        let func: Function = "3^2^3".parse().unwrap();
        assert_eq!(func.eval(0.0).unwrap(), 6561.0);
    }

    #[test]
    fn existing() {
        let func1: Function = "1 + x".parse().unwrap();
        let func1 = Rc::new(func1);
        let func2 = Function::BinaryOp(Box::new(Function::Variable), BinaryOperator::Plus, Box::new(Function::Existing(func1.clone())));
        println!("{}", func2);
        assert_eq!(5.0, func2.eval(2.0).unwrap());
    }
}
