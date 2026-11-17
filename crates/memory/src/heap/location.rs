// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

//! Slot addresses: [`HeapLoc`] for the whole heap and [`PageLoc`] for a
//! single page.

/// A slot's address in the heap, packed into 32 bits from the most
/// significant bit as `page(20) : chunk(6) : slot(6)`. This is the
/// location field of a [`Handle`](crate::Handle).
///
/// Construction checks every field, and the accessors mask, so a
/// `HeapLoc` always names a slot that could exist. It does not mean the
/// page is registered or the slot occupied.
///
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HeapLoc(u32);

impl HeapLoc {

    /// Number of bits addressing a page within the heap directory.
    pub const PAGE_BITS: u32 = 20;

    /// Number of pages addressable in the heap directory.
    pub const PAGE_COUNT: u32 = 1 << HeapLoc::PAGE_BITS;

    /// Mask selecting a page index once shifted down, `PAGE_COUNT - 1`.
    pub const PAGE_MASK: u32 = HeapLoc::PAGE_COUNT - 1;

    /// `PAGE_COUNT` as a usize, for ease of use.
    pub const PAGE_COUNT_SZ: usize = HeapLoc::PAGE_COUNT as usize;

    /// Number of bits addressing a slot within a chunk, and a chunk
    /// within a page.
    pub const ENTRY_BITS: u32 = 6;

    /// Number of slots per chunk, and of chunks per page.
    pub const ENTRY_COUNT: u32 = 1 << HeapLoc::ENTRY_BITS;

    /// Mask selecting a chunk or slot index once shifted down,
    /// `ENTRY_COUNT - 1`.
    pub const ENTRY_MASK:  u32 = HeapLoc::ENTRY_COUNT - 1;

    /// `ENTRY_COUNT` as a usize, for ease of use.
    pub const ENTRY_COUNT_SZ: usize = HeapLoc::ENTRY_COUNT as usize;
}

const _: () = assert!(
    HeapLoc::ENTRY_BITS + HeapLoc::ENTRY_BITS + HeapLoc::PAGE_BITS == 32,
    "expected heap location bits to add up to 32"
);


impl HeapLoc {

    /// Creates a heap location from a page, chunk and slot.
    ///
    /// # Panics
    ///
    /// If `page` is not less than `PAGE_COUNT`, or if either `chunk` or
    /// `slot` is not less than `ENTRY_COUNT`.
    ///
    #[must_use]
    pub const fn new(
        page:  u32,
        chunk: u32,
        slot:  u32
    ) -> Self {
        assert!(page  < HeapLoc::PAGE_COUNT, "page must be less than PAGE_COUNT");
        assert!(chunk < HeapLoc::ENTRY_COUNT, "chunk must be less than ENTRY_COUNT");
        assert!(slot  < HeapLoc::ENTRY_COUNT, "slot must be less than ENTRY_COUNT");
        Self((page << (2*HeapLoc::ENTRY_BITS)) | (chunk << HeapLoc::ENTRY_BITS) | slot)
    }

}

impl HeapLoc {

    /// This location with its page replaced.
    ///
    /// # Panics
    ///
    /// If `page` is not less than `PAGE_COUNT`.
    ///
    #[must_use]
    pub const fn with_page(self, page: u32) -> Self {
        Self::new(page, self.chunk(), self.slot())
    }

    /// This location with its chunk replaced.
    ///
    /// # Panics
    ///
    /// If `chunk` is not less than `ENTRY_COUNT`.
    ///
    #[must_use]
    pub const fn with_chunk(self, chunk: u32) -> Self {
        Self::new(self.page(), chunk, self.slot())
    }

    /// This location with its slot replaced.
    ///
    /// # Panics
    ///
    /// If `slot` is not less than `ENTRY_COUNT`.
    ///
    #[must_use]
    pub const fn with_slot(self, slot: u32) -> Self {
        Self::new(self.page(), self.chunk(), slot)
    }

}

impl HeapLoc {

    /// The index of the page in the heap directory. Always less than
    /// `PAGE_COUNT`.
    #[must_use]
    pub const fn page(self) -> u32 {
        (self.0 >> (2*HeapLoc::ENTRY_BITS)) & HeapLoc::PAGE_MASK
    }

    /// The index of the chunk in its page. Always less than
    /// `ENTRY_COUNT`.
    #[must_use]
    pub const fn chunk(self) -> u32 {
        (self.0 >> HeapLoc::ENTRY_BITS) & HeapLoc::ENTRY_MASK
    }

    /// The index of the slot in its chunk. Always less than
    /// `ENTRY_COUNT`.
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.0 & HeapLoc::ENTRY_MASK
    }

}

impl HeapLoc {

    /// `page` as a usize, for indexing.
    #[must_use]
    pub const fn page_sz(self) -> usize {
        self.page() as usize
    }

    /// `chunk` as a usize, for indexing.
    #[must_use]
    pub const fn chunk_sz(self) -> usize {
        self.chunk() as usize
    }

    /// `slot` as a usize, for indexing.
    #[must_use]
    pub const fn slot_sz(self) -> usize {
        self.slot() as usize
    }

}

impl core::fmt::Debug for HeapLoc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "HeapLoc({}:{}:{})", self.page(), self.chunk(), self.slot())
    }
}

/// A slot's address within a single page: the chunk and slot of a
/// [`HeapLoc`], without the page.
///
/// Pages take and return `PageLoc`s, so a page can't be handed another
/// page's address or return one with a page id it doesn't know. The
/// owner of the pages converts with `from_heap_loc` and `into_heap_loc`.
/// It wraps a `HeapLoc` with page 0, so both conversions are a mask.
///
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct PageLoc(HeapLoc);

impl PageLoc {

    /// Creates a page location from a chunk and slot.
    ///
    /// # Panics
    ///
    /// If either `chunk` or `slot` is not less than
    /// `HeapLoc::ENTRY_COUNT`.
    ///
    #[must_use]
    pub const fn new(
        chunk: u32,
        slot:  u32
    ) -> Self {
        Self(HeapLoc::new(0, chunk, slot))
    }

    /// The heap location of this slot on page `page`.
    ///
    /// # Panics
    ///
    /// If `page` is not less than `HeapLoc::PAGE_COUNT`.
    ///
    #[must_use]
    pub const fn into_heap_loc(self, page: u32) -> HeapLoc {
        self.0.with_page(page)
    }

    /// The chunk and slot of `heap_loc`, discarding its page.
    #[must_use]
    pub const fn from_heap_loc(heap_loc: HeapLoc) -> Self {
        Self(heap_loc.with_page(0))
    }

}

impl PageLoc {

    /// This location with its chunk replaced.
    ///
    /// # Panics
    ///
    /// If `chunk` is not less than `HeapLoc::ENTRY_COUNT`.
    ///
    #[must_use]
    pub const fn with_chunk(self, chunk: u32) -> Self {
        Self(self.0.with_chunk(chunk))
    }

    /// This location with its slot replaced.
    ///
    /// # Panics
    ///
    /// If `slot` is not less than `HeapLoc::ENTRY_COUNT`.
    ///
    #[must_use]
    pub const fn with_slot(self, slot: u32) -> Self {
        Self(self.0.with_slot(slot))
    }

}

impl PageLoc {

    /// The index of the chunk in the page. Always less than
    /// `HeapLoc::ENTRY_COUNT`.
    #[must_use]
    pub const fn chunk(self) -> u32 {
        self.0.chunk()
    }

    /// The index of the slot in its chunk. Always less than
    /// `HeapLoc::ENTRY_COUNT`.
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.0.slot()
    }

}

impl PageLoc {

    /// `chunk` as a usize, for indexing.
    #[must_use]
    pub const fn chunk_sz(self) -> usize {
        self.0.chunk() as usize
    }

    /// `slot` as a usize, for indexing.
    #[must_use]
    pub const fn slot_sz(self) -> usize {
        self.0.slot() as usize
    }

}

impl core::fmt::Debug for PageLoc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PageLoc({}:{})", self.chunk(), self.slot())
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

    /// Fields survive packing at their maximums, and each `with_*`
    /// replaces only its own field.
    #[test]
    fn fields_round_trip() {
        let max = HeapLoc::new(HeapLoc::PAGE_MASK, HeapLoc::ENTRY_MASK, HeapLoc::ENTRY_MASK);
        assert_eq!(max.page(),  HeapLoc::PAGE_MASK);
        assert_eq!(max.chunk(), HeapLoc::ENTRY_MASK);
        assert_eq!(max.slot(),  HeapLoc::ENTRY_MASK);

        let loc = HeapLoc::new(5, 6, 7);
        assert_eq!(loc.with_page(9),  HeapLoc::new(9, 6, 7));
        assert_eq!(loc.with_chunk(9), HeapLoc::new(5, 9, 7));
        assert_eq!(loc.with_slot(9),  HeapLoc::new(5, 6, 9));
    }

    /// Converting to a page location discards the page, so the same slot
    /// on different pages compares equal once converted, and converting
    /// back stamps the given page.
    #[test]
    fn page_loc_discards_page() {
        let a = PageLoc::from_heap_loc(HeapLoc::new(3, 4, 5));
        let b = PageLoc::from_heap_loc(HeapLoc::new(8, 4, 5));
        assert_eq!(a, PageLoc::new(4, 5));
        assert_eq!(a, b);
        assert_eq!(a.into_heap_loc(8), HeapLoc::new(8, 4, 5));
    }

    /// A page index past the directory panics.
    #[test]
    #[should_panic(expected = "page must be less than PAGE_COUNT")]
    fn new_rejects_page_past_directory() {
        let _ = HeapLoc::new(HeapLoc::PAGE_COUNT, 0, 0);
    }
}
