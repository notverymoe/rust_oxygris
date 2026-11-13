// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

//! The size-class heap: pages of fixed-size slots, the GC phase that
//! drives them, and the layout descriptions the write barrier uses to
//! find handles.
//! 
use crate::{bitmap::BitMap64, heap::page::{PAGE_ENTRY_BITS, PAGE_ENTRY_MASK}};

pub mod page;

// TODO impl Error
/// Errors returned when an allocation cannot be satisfied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    /// Every slot in the page is occupied, even after catching the page
    /// up to the GC's sweep target, and no further chunks can be
    /// committed.
    PageFull,
}

/// The number of size classes: 16 linear classes of 16 B up to 256 B,
/// then 8 doublings up to 64 KiB.
pub const CLASS_COUNT: u32 = 24;

/// Bytes held by `class`: 16-byte steps to 256, then doublings to 64 KiB.
/// TODO move elsewhere
#[must_use]
pub const fn class_bytes(class: u32) -> u32 {
    if class < 16 { 
        (class + 1) << 4 
    } else { 
        256 << (class - 15) 
    }
}

/// The smallest class whose slot holds `bytes`.
/// TODO move elsewhere
#[must_use]
pub const fn class_for(bytes: u32) -> u32 {
    if bytes <= 256 {
        bytes.div_ceil(16).saturating_sub(1)
    } else {
        15 + (bytes - 1).ilog2() - 7
    }
}

/// Describes which handle-words of a slot's layout hold handles, so the
/// write barrier can find them without knowing the object's type.
/// Offsets are in handle-words from the start of the slot.
///
#[derive(Debug, Clone, Copy)]
pub enum HandleMap<'a> {
    /// The first `n` handle-words of the slot are handles and no others
    /// are. The common layout, with handle fields sorted first.
    Scan(u32),

    /// One bit per handle-word, set where the word holds a handle. Bit
    /// `i % 64` of word `i / 64` describes handle-word `i`. Words past
    /// the end of the slice read as holding no handles.
    Map(&'a [BitMap64]),
}

impl HandleMap<'_> {

    /// Whether the handle-word at `handle_hw` holds a handle in this
    /// layout. Offsets past the end of the layout report false.
    #[must_use]
    pub const fn is_handle(&self, handle_hw: u32) -> bool {
        match *self {
            HandleMap::Scan(count) => handle_hw < count,
            HandleMap::Map(map) => {
                let word = (handle_hw >> PAGE_ENTRY_BITS) as usize;
                word < map.len() && map[word].get(handle_hw & PAGE_ENTRY_MASK)
            },
        }
    }

}

/// The GC phase as seen by a page, passed in on each call that
/// depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcPhaseInfo {
    /// The sweep index fed by the most recently started marking phase.
    /// At least 1 while `marking` is true.
    mark_epoch: u64,

    /// Whether a marking phase is in progress, which enables allocating
    /// black and the write barrier.
    marking: bool,
}

impl GcPhaseInfo {

    /// Creates a snapshot of the information required for accurate
    /// GC marking and handle dirtying. 
    /// 
    /// # Panics
    /// 
    /// Will panic if `mark_epoch` is `0` when `marking` is `true`.
    /// 
    #[must_use]
    pub const fn new(
        mark_epoch: u64,
        marking: bool,
    ) -> Self {
        assert!(!marking || mark_epoch > 0, "mark_epoch must be at least 1 while marking");
        Self { mark_epoch, marking }
    }

    /// The furthest any page may be swept right now. While marking, the
    /// sweep for `mark_epoch` depends on incomplete marks, so pages may
    /// only be brought up to `mark_epoch - 1`. Otherwise pages may be
    /// brought up to `mark_epoch`.
    ///
    /// This is the single definition of the target. New pages should be
    /// created at it, and background sweeping should pass it to
    /// `Page::sweep_to`.
    /// 
    #[must_use]
    pub const fn sweep_target(self) -> u64 {
        if self.marking { self.mark_epoch - 1 } else { self.mark_epoch }
    }

    /// The sweep index that the most recently started marking phase
    /// feeds. Must be at least 1 while `marking` is true.
    #[must_use]
    pub const fn mark_epoch(self) -> u64 {
        self.mark_epoch
    }

    /// Whether a marking phase is currently in progress.
    #[must_use]
    pub const fn marking(self) -> bool {
        self.marking
    }
}