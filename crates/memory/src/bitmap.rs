// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

//! A fixed-width bitmap and the run-walking and summary operations the
//! page builds its slot and chunk state from.
//!
//! [`BitMap64`] is a `u64` where bit `i` stands for entry `i`. The page
//! keeps one per chunk for occupancy, marks and dirty slots, and one per
//! page summarizing each of those across chunks, so the map's width is
//! also the number of slots in a chunk and of chunks in a page.
//!
//! Beyond the bitwise operators, the map supports lowest-first search
//! through [`BitMap64::trailing_ones`] and [`BitMap64::trailing_zeros`],
//! walking a map as alternating runs through [`BitMap64::consume_zeros`]
//! and [`BitMap64::consume_ones`], and folding a slice of maps into one
//! bit per map through the `summarize_*` constructors.

use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// A 64-bit bitmap where bit `i` addresses entry `i`. Used for per-chunk
/// slot state and per-page chunk summaries, so its width matches
/// `PAGE_ENTRY_COUNT`.
///
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BitMap64(u64);

impl Not for BitMap64 {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl BitAnd for BitMap64 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitOr for BitMap64 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitXor for BitMap64 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitAndAssign for BitMap64 {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOrAssign for BitMap64 {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXorAssign for BitMap64 {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl BitMap64 {

    /// Every bit set.
    pub const ALL:  Self = Self(u64::MAX);

    /// No bits set.
    pub const NONE: Self = Self(0);

    /// The raw bits.
    #[inline]
    #[must_use]
    pub const fn into_inner(self) -> u64 {
        self.0
    }

    /// A run of `len` set bits starting at bit `from`. A `len` of 64 or
    /// more fills from `from` up to the top bit.
    ///
    /// `from` must be less than 64. Bits that would extend past the top
    /// are discarded.
    ///
    #[inline]
    #[must_use]
    pub const fn span(from: u32, len: u32) -> Self {
        let ones = if len >= 64 { u64::MAX } else { (1_u64 << len) - 1 };
        Self(ones << from)
    }

    /// Sets bit `idx` to `value`, returning whether the bit changed.
    ///
    /// Callers rely on the return value to count or detect transitions,
    /// such as newly marked or newly dirtied slots. `idx` must be less
    /// than 64.
    ///
    #[inline]
    pub const fn set(&mut self, idx: u32, value: bool) -> bool {
        let old = self.0;
        self.0 = (self.0 & !(1 << idx)) | ((value as u64) << idx);
        self.0 != old
    }

    /// Whether bit `idx` is set. `idx` must be less than 64.
    #[inline]
    #[must_use]
    pub const fn get(self, idx: u32) -> bool {
        (self.0 >> idx) & 0x01 == 0x01
    }

    /// Sets bit `idx`, returning whether it was previously clear.
    #[inline]
    pub const fn mark(&mut self, idx: u32) -> bool {
        self.set(idx, true)
    }

    /// Clears bit `idx`, returning whether it was previously set.
    #[inline]
    pub const fn clear(&mut self, idx: u32) -> bool {
        self.set(idx, false)
    }

}

impl BitMap64 {
    /// The number of consecutive set bits from bit 0: the index of the
    /// lowest clear bit, or 64 if every bit is set.
    #[inline]
    #[must_use]
    pub const fn trailing_ones(self) -> u32 {
        self.0.trailing_ones()
    }

    /// The number of consecutive clear bits from bit 0: the index of the
    /// lowest set bit, or 64 if no bit is set.
    #[inline]
    #[must_use]
    pub const fn trailing_zeros(self) -> u32 {
        self.0.trailing_zeros()
    }

    /// The number of consecutive set bits from bit 63 downwards.
    #[inline]
    #[must_use]
    pub const fn leading_ones(self) -> u32 {
        self.0.leading_ones()
    }

    /// The number of consecutive clear bits from bit 63 downwards.
    #[inline]
    #[must_use]
    pub const fn leading_zeros(self) -> u32 {
        self.0.leading_zeros()
    }

    /// The number of set bits.
    #[inline]
    #[must_use]
    pub const fn count_ones(self) -> u32 {
        self.0.count_ones()
    }

    /// The number of clear bits.
    #[inline]
    #[must_use]
    pub const fn count_zeros(self) -> u32 {
        self.0.count_zeros()
    }

    /// Removes the run of clear bits at the bottom of the map, shifting
    /// the rest down, and returns the run's length. Returns 64 and
    /// leaves the map empty if no bits were set.
    ///
    /// Alternating with `consume_ones` walks the map as runs. The sum
    /// of the returned lengths is the index of the next unvisited bit.
    ///
    #[inline]
    pub const fn consume_zeros(&mut self) -> u32 {
        let count = self.0.trailing_zeros();
        self.0 = if count < u64::BITS { self.0 >> count } else { 0 };
        count
    }

    /// Removes the run of set bits at the bottom of the map, shifting the
    /// rest down, and returns the run's length. Returns 64 and leaves
    /// the map empty if every bit was set, or 0 if bit 0 was clear.
    ///
    #[inline]
    pub const fn consume_ones(&mut self) -> u32 {
        let count = self.0.trailing_ones();
        self.0 = if count < u64::BITS { self.0 >> count } else { 0 };
        count
    }

}

impl BitMap64 {

    /// Whether every bit is set.
    #[inline]
    #[must_use]
    pub const fn all(self) -> bool {
        self.0 == u64::MAX
    }

    /// Whether at least one bit is set.
    #[inline]
    #[must_use]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// Whether at least one bit is clear: the negation of `all`.
    #[inline]
    #[must_use]
    pub const fn nall(self) -> bool {
        self.0 != u64::MAX
    }

    /// Whether no bits are set.
    #[inline]
    #[must_use]
    pub const fn none(self) -> bool {
        self.0 == 0
    }
}

impl BitMap64 {
    /// Builds a summary bitmap from `maps`, setting bit `base + i` to
    /// `f(maps[i])`. Bits outside the covered range keep their value
    /// from `fill`, and maps that would land past bit 63 are ignored.
    /// `base` must be at most 64.
    ///
    fn summarize(f: impl Fn(Self) -> bool, fill: Self, base: u32, maps: &[Self]) -> Self {
        let mut result = fill;
        for (i, &m) in maps.iter().take(64 - base as usize).enumerate() {
            result.set(base + i as u32, f(m));
        }
        result
    }

    /// Summarizes `maps` with a bit set for each map that is entirely
    /// set. See `summarize`.
    /// 
    pub fn summarize_all(fill: Self, base: u32, maps: &[Self])  -> Self { 
        Self::summarize(Self::all, fill, base, maps) 
    }

    /// Summarizes `maps` with a bit set for each map with any bit set.
    /// See `summarize`.
    /// 
    pub fn summarize_any(fill: Self, base: u32, maps: &[Self])  -> Self { 
        Self::summarize(Self::any, fill, base, maps) 
    }

    /// Summarizes `maps` with a bit set for each map with any bit clear.
    /// See `summarize`.
    /// 
    pub fn summarize_nall(fill: Self, base: u32, maps: &[Self]) -> Self { 
        Self::summarize(Self::nall, fill, base, maps) 
    }

    /// Summarizes `maps` with a bit set for each map with no bits set.
    /// See `summarize`.
    /// 
    pub fn summarize_none(fill: Self, base: u32, maps: &[Self]) -> Self { 
        Self::summarize(Self::none, fill, base, maps)
    }
}