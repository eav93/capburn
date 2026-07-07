//! Configurable captcha charset: character <-> class-index mapping.
//!
//! A charset is given as a spec string. Tokens are separated by `+` and each
//! token is either the name of a built-in set or a literal sequence of
//! characters. A charset can also be detected automatically from a dataset
//! ([`Charset::from_observed`]).
//!
//! Spec examples:
//! - `digits`            — digits `0-9` only
//! - `cyrillic+digits`   — lowercase Cyrillic and digits
//! - `upper+digits`      — uppercase Latin letters and digits
//! - `ABCDEF0123456789`  — exactly those characters (a hex set)

use std::collections::HashMap;

/// Lowercase Cyrillic (33 letters, including "ё").
pub const CYRILLIC_LOWER: &str = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
/// Uppercase Cyrillic (33 letters, including "Ё").
pub const CYRILLIC_UPPER: &str = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
pub const LATIN_LOWER: &str = "abcdefghijklmnopqrstuvwxyz";
pub const LATIN_UPPER: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const DIGITS: &str = "0123456789";

/// Character families recognized by auto-detection. The order here defines the
/// class order in the resulting charset.
const FAMILIES: [&str; 5] = [
    DIGITS,
    LATIN_LOWER,
    LATIN_UPPER,
    CYRILLIC_LOWER,
    CYRILLIC_UPPER,
];

/// Alphabet of recognizable characters.
///
/// The character order defines the class index and must stay stable between
/// training and inference — that is why the expanded character string
/// ([`Charset::as_chars`]) is what gets saved to the config, not the original
/// spec.
#[derive(Clone, Debug)]
pub struct Charset {
    chars: Vec<char>,
    index: HashMap<char, usize>,
}

impl Charset {
    /// Parse a spec like `digits`, `cyrillic+digits`, `ABC012`.
    pub fn from_spec(spec: &str) -> Result<Self, String> {
        let mut chars: Vec<char> = Vec::new();
        for token in spec.split('+') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let expanded = expand_named(token).unwrap_or(token);
            for c in expanded.chars() {
                if !chars.contains(&c) {
                    chars.push(c);
                }
            }
        }
        if chars.is_empty() {
            return Err(format!("empty charset from spec: {spec:?}"));
        }
        Ok(Self::from_chars_vec(chars))
    }

    /// Build a charset from an already-expanded character string (config load).
    pub fn from_chars(chars: &str) -> Self {
        let mut unique: Vec<char> = Vec::new();
        for c in chars.chars() {
            if !unique.contains(&c) {
                unique.push(c);
            }
        }
        Self::from_chars_vec(unique)
    }

    /// Auto-detect the charset from the characters observed in labels.
    ///
    /// Works "by family": if at least one character of a family (digits,
    /// lower/upper Latin, lower/upper Cyrillic) is seen, the whole family is
    /// included. In other words, one Russian letter in the dataset means "all
    /// Russian letters", one English letter means "all English letters".
    pub fn from_observed<I: IntoIterator<Item = char>>(observed: I) -> Result<Self, String> {
        let mut present = [false; FAMILIES.len()];
        for c in observed {
            for (i, fam) in FAMILIES.iter().enumerate() {
                if fam.contains(c) {
                    present[i] = true;
                }
            }
        }
        let mut chars: Vec<char> = Vec::new();
        for (i, fam) in FAMILIES.iter().enumerate() {
            if present[i] {
                chars.extend(fam.chars());
            }
        }
        if chars.is_empty() {
            return Err("could not detect charset: no known characters in labels".into());
        }
        Ok(Self::from_chars_vec(chars))
    }

    /// Human-readable list of character families included in the charset.
    pub fn describe_families(&self) -> String {
        let mut names = Vec::new();
        if self.chars.iter().any(|c| DIGITS.contains(*c)) {
            names.push("digits");
        }
        if self.chars.iter().any(|c| LATIN_LOWER.contains(*c)) {
            names.push("latin lowercase");
        }
        if self.chars.iter().any(|c| LATIN_UPPER.contains(*c)) {
            names.push("latin uppercase");
        }
        if self.chars.iter().any(|c| CYRILLIC_LOWER.contains(*c)) {
            names.push("cyrillic lowercase");
        }
        if self.chars.iter().any(|c| CYRILLIC_UPPER.contains(*c)) {
            names.push("cyrillic uppercase");
        }
        if names.is_empty() {
            "custom set".into()
        } else {
            names.join(", ")
        }
    }

    fn from_chars_vec(chars: Vec<char>) -> Self {
        let index = chars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        Self { chars, index }
    }

    /// Number of classes (charset size).
    pub fn len(&self) -> usize {
        self.chars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Class index for a character, if it belongs to the charset.
    pub fn index_of(&self, c: char) -> Option<usize> {
        self.index.get(&c).copied()
    }

    /// Character for a class index.
    pub fn char_at(&self, i: usize) -> char {
        self.chars[i]
    }

    /// Character for a class index without panicking (for untrusted input or a
    /// corrupt model).
    pub fn try_char_at(&self, i: usize) -> Option<char> {
        self.chars.get(i).copied()
    }

    /// Expanded character string — saved to the config for inference.
    pub fn as_chars(&self) -> String {
        self.chars.iter().collect()
    }
}

/// Expand the name of a built-in character set.
fn expand_named(name: &str) -> Option<&'static str> {
    match name.to_ascii_lowercase().as_str() {
        "digits" | "digit" | "num" | "numeric" => Some(DIGITS),
        "lower" | "latin_lower" | "lowercase" => Some(LATIN_LOWER),
        "upper" | "latin_upper" | "uppercase" => Some(LATIN_UPPER),
        "letters" | "latin" => Some("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "cyrillic" | "cyrillic_lower" => Some(CYRILLIC_LOWER),
        "cyrillic_upper" => Some(CYRILLIC_UPPER),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_spec_digits() {
        let cs = Charset::from_spec("digits").unwrap();
        assert_eq!(cs.len(), 10);
        assert_eq!(cs.index_of('0'), Some(0));
        assert_eq!(cs.index_of('9'), Some(9));
        assert_eq!(cs.index_of('a'), None);
    }

    #[test]
    fn from_spec_combination_dedups() {
        let cs = Charset::from_spec("digits+0123").unwrap();
        assert_eq!(cs.len(), 10, "duplicate characters must not be repeated");
    }

    #[test]
    fn from_spec_cyrillic_plus_digits() {
        let cs = Charset::from_spec("cyrillic+digits").unwrap();
        assert_eq!(cs.len(), 33 + 10);
        assert_eq!(cs.index_of('а'), Some(0));
        assert!(cs.index_of('5').is_some());
    }

    #[test]
    fn from_spec_literal() {
        let cs = Charset::from_spec("ABC012").unwrap();
        assert_eq!(cs.len(), 6);
        assert_eq!(cs.as_chars(), "ABC012");
    }

    #[test]
    fn from_chars_dedups() {
        let cs = Charset::from_chars("aabbcc");
        assert_eq!(cs.as_chars(), "abc");
    }

    #[test]
    fn from_observed_by_family() {
        // one digit and one Russian letter → all digits + all lowercase Cyrillic
        let cs = Charset::from_observed("7я".chars()).unwrap();
        assert_eq!(cs.len(), 10 + 33);
        assert!(cs.index_of('0').is_some());
        assert!(cs.index_of('а').is_some());
        assert!(cs.index_of('A').is_none());
    }

    #[test]
    fn try_char_at_out_of_range() {
        let cs = Charset::from_spec("digits").unwrap();
        assert_eq!(cs.try_char_at(0), Some('0'));
        assert_eq!(cs.try_char_at(999), None);
    }

    #[test]
    fn empty_spec_errors() {
        assert!(Charset::from_spec("").is_err());
        assert!(Charset::from_spec("+").is_err());
    }
}
