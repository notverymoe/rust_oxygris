// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

//! The slot size-class table.
//!
//! Classes 0 to 15 step linearly by 16 B, from 16 B to 256 B. Classes 16
//! to 23 double, from 512 B to 64 KiB. There is no 8-byte class, so the
//! smallest slot is 16 B, and every slot size is a whole number of
//! handle-words.

/// The number of size classes: 16 linear classes of 16 B up to 256 B,
/// then 8 doublings up to 64 KiB.
pub const CLASS_COUNT: u32 = 24;

/// `CLASS_COUNT` as a usize, for ease of use.
pub const CLASS_COUNT_SZ: usize = CLASS_COUNT as usize;

/// The slot size of the largest class in bytes, 64 KiB. Objects larger
/// than this have no class and cannot be allocated on a page.
pub const CLASS_COUNT_MAX_SIZE: u32 = size_of_class(CLASS_COUNT-1).unwrap();

/// The slot size in bytes of `class`: 16-byte steps to 256 B, then
/// doublings to 64 KiB.
///
/// Returns `None` if `class` is not less than `CLASS_COUNT`.
#[must_use]
pub const fn size_of_class(class: u32) -> Option<u32> {
    if class < 16 {
        Some((class + 1) << 4)
    } else if class < CLASS_COUNT {
        Some(256 << (class - 15))
    } else {
        None
    }
}

/// The smallest class whose slots hold `size` bytes. A `size` of 0 maps
/// to class 0.
///
/// Returns `None` if `size` exceeds `CLASS_COUNT_MAX_SIZE`.
#[must_use]
pub const fn class_for_size(size: u32) -> Option<u32> {
    if size <= 256 {
        Some(size.div_ceil(16).saturating_sub(1))
    } else if size <= CLASS_COUNT_MAX_SIZE {
        Some(15 + (size - 1).ilog2() - 7)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    //! **Disclosure** Unless otherwise stated on the test, these tests
    //!     were written by an LLM and were not reviewed by a human.

    #![allow(
        clippy::missing_panics_doc,
        reason = "tests and their helpers panic by design; the assertion messages are the documentation"
    )]

    use super::*;

    /// Classes past the table and sizes past the largest class have no
    /// answer, while the largest class's own size still maps to it.
    #[test]
    fn out_of_range_is_none() {
        assert_eq!(size_of_class(CLASS_COUNT), None);
        assert_eq!(size_of_class(u32::MAX), None);
        assert_eq!(class_for_size(CLASS_COUNT_MAX_SIZE), Some(CLASS_COUNT - 1));
        assert_eq!(class_for_size(CLASS_COUNT_MAX_SIZE + 1), None);
        assert_eq!(class_for_size(u32::MAX), None);
    }

}