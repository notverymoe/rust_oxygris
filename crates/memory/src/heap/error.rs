// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

/// Errors returned when an allocation cannot be satisfied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeapAllocError {
    /// Every slot in the page is occupied, even after catching the page
    /// up to the GC's sweep target, and no further chunks can be
    /// committed.
    PageFull,

    /// Every page of the size class is full, even after a full
    /// collection cycle, and no further pages can be committed.
    HeapFull,
}
