pub fn estimate_tokens(input: &str) -> u32 {
    let ascii_chars = input
        .chars()
        .filter(|character| character.is_ascii())
        .count();
    let non_ascii_chars = input
        .chars()
        .filter(|character| !character.is_ascii())
        .count();
    let by_chars = ascii_chars.div_ceil(4).saturating_add(non_ascii_chars);
    let by_words = input
        .split_whitespace()
        .count()
        .saturating_mul(4)
        .div_ceil(3);
    let estimate = by_chars.max(by_words).max(1);
    estimate.min(u32::MAX as usize) as u32
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

    #[test]
    fn estimates_non_ascii_text_conservatively() {
        let input = "你好世界";
        assert!(estimate_tokens(input) >= input.chars().count() as u32);
    }
}
