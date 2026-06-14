pub fn estimate_tokens(input: &str) -> u32 {
    let by_chars = input.chars().count().div_ceil(4);
    let by_words = input.split_whitespace().count() * 4 / 3;
    by_chars.max(by_words).max(1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_nonzero_tokens() {
        assert_eq!(estimate_tokens(""), 1);
        assert!(estimate_tokens("hello world") >= 2);
        assert!(estimate_tokens(&"a".repeat(400)) >= 100);
    }
}
