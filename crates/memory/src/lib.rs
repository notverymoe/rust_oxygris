// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

//! Memory management for the Oxygris virtual machine: a size-class paged
//! heap with an incremental, per-page garbage collector.
//!
//! Objects live in fixed-size slots on a [`heap::page::Page`]. A page
//! holds up to 64 chunks of 64 slots, commits chunks lazily and in
//! order, and always hands out its lowest free slot. Objects are
//! addressed by a [`Handle`], a structured 64-bit word whose location
//! bits name the page, chunk and slot.
//!
//! Collection is an incremental mark and sweep. Each page carries its
//! own sweep index, and its live mark is the parity of that index, so
//! survivors read as unmarked after a sweep without a clearing pass.
//! Pages may lag behind the collector and are caught up lazily by the
//! first allocation or mark that touches them; a page no marking has
//! reached for a full cycle is reset wholesale. Mutation during marking
//! goes through an insertion write barrier that records dirty
//! handle-words at field granularity, which the marker drains before
//! finishing.
//!
//! The crate is `no_std` with `alloc`, has no dependencies, and forbids
//! unsafe code. It is the lowest layer of the VM and knows nothing about
//! types or values beyond which handle-words of a slot hold handles.
//!
//! # Modules
//!
//! - [`bitmap`]: the 64-bit bitmap used for per-chunk slot state and
//!   per-page chunk summaries.
//! - [`heap`]: the size-class table, the GC phase as a page sees it, and
//!   the layout descriptions the write barrier uses to find handles.
//! - [`heap::page`]: the page itself.
//!
//! The heap that owns pages, resolves handles to locations and drives
//! the collector is not yet written.

#![forbid(unsafe_code)]
#![no_std]
extern crate alloc;

pub mod bitmap;

// WIP
pub mod heap;


// TODO finish handle impl
/// A structured 64-bit reference word, laid out from the most
/// significant bit as `kind(4) : reserved(4) : scan(4) : type(20) :
/// location(32)`, where location is `page(20) : chunk(6) : slot(6)`.
/// Handles and refs share this format and are told apart by `kind`.
///
/// Kind 0 is null, and the all-zero word is `NONE`, so zeroed slot
/// memory always reads as null.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Handle(u64);

impl Handle {

    /// Mask selecting the `kind` bits, the top 4 bits of the word.
    pub const KIND_MASK: u64 =  0xF << 60;

    /// The canonical null handle: the all-zero word. Any word of kind 0
    /// is null, so test with `is_none` rather than comparing to `NONE`.
    /// Never traced, and ignored by the write barrier.
    ///
    pub const NONE: Self = Self(0);

    /// The size of a handle in bytes, and so of a handle-word.
    pub const BYTES: usize = (u64::BITS >> 3) as usize;
    
    /// Reads a handle from its in-memory representation in slot bytes.
    #[must_use]
    pub const fn from_ne_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_ne_bytes(bytes))
    }
    
    /// The in-memory representation of the handle, for writing into slot
    /// bytes.
    #[must_use]
    pub const fn to_ne_bytes(self) -> [u8; 8] {
        u64::to_ne_bytes(self.0)
    }

    /// Whether this is a null handle, meaning its kind is 0. The other
    /// bits are ignored, so this can be true for words other than `NONE`.
    #[must_use]
    pub const fn is_none(self) -> bool {
        (self.0 & Self::KIND_MASK) == 0
    }

}