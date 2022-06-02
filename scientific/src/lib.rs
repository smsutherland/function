#[cfg(test)]
mod tests {
    use function::E;
    #[test]
    fn it_works() {
        let result = E;
        assert_eq!(result, std::f64::consts::E);
    }
}
