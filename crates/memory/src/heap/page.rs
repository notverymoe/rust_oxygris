// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

//! Implementation of a size-class memory page. Allocates chunks of
//! slots of a fixed size-class of byte size, always claiming and
//! reusing the lowest free slot first.
//!
//! The page tracks its own GC state through a per-page sweep index.
//! Its live mark is the parity of that index, so the meaning of mark
//! bits flips each time the page is swept and survivors become
//! unmarked without a clearing pass. Pages may lag behind the GC's
//! sweep target; marking and allocation catch them up lazily. The
//! GC phase is passed in on each call that needs it and is never
//! stored by the page.

use core::ops::Range;

use alloc::{boxed::Box, vec};

use crate::{
    Handle, bitmap::BitMap64, heap::{AllocError, CLASS_COUNT, GcPhaseInfo, HandleMap, class_bytes}
};

/// Log2 of the size of a handle-word in bytes. Handle fields are
/// aligned to handle-words, and dirty tracking is indexed in them.
pub const HANDLE_ALIGNMENT_BITS: u32 = 3;

const _: () = assert!(
    1 << HANDLE_ALIGNMENT_BITS == Handle::BYTES,
    "`HANDLE_ALIGNMENT_BITS` must match `Handle::BYTES`",
);

/// Number of bits used to address a slot within a chunk, and a chunk
/// within a page.
pub const PAGE_ENTRY_BITS:  u32 = 6;

/// Number of slots per chunk, and of chunks per page.
pub const PAGE_ENTRY_COUNT: u32 = 1 << PAGE_ENTRY_BITS;

/// Mask selecting the bits addressable by `PAGE_ENTRY_BITS`.
pub const PAGE_ENTRY_MASK:  u32 = PAGE_ENTRY_COUNT - 1;

/// `PAGE_ENTRY_COUNT` as a usize, for ease of use.
pub const PAGE_ENTRY_COUNT_SZ: usize = PAGE_ENTRY_COUNT as usize;

const _: () = assert!(
    PAGE_ENTRY_COUNT == u64::BITS,
    "a chunk's dirty map is exactly `slot_hw` words only when slots per chunk equals bits per word",
);

/// The location of a slot within a page: a chunk index and a slot
/// index within that chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageLocation {
    /// The index of the chunk in the page. Must be less than
    /// `PAGE_ENTRY_COUNT`; out-of-range locations read as unoccupied.
    pub chunk: u32,

    /// The index of the slot in the chunk. Must be less than
    /// `PAGE_ENTRY_COUNT`; out-of-range locations read as unoccupied.
    pub slot: u32,
}

/// The result of a successful `Page::alloc`.
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct AllocInfo {
    /// The location of the new allocation.
    pub location: PageLocation,

    /// Whether initialization newly marked the page dirty. Only
    /// possible when allocated during marking.
    pub was_dirtied: bool,

    /// Whether this allocation made the page completely occupied.
    pub became_full: bool,

    /// The catch-up sweep or reset performed before allocating, if any.
    /// The caller must apply it to its accounting before the other
    /// fields, since `became_full` describes the page after the sweep.
    ///
    pub swept: Option<SweepInfo>,
}

/// A size-class page of up to `PAGE_ENTRY_COUNT` chunks, each holding
/// `PAGE_ENTRY_COUNT` fixed-size slots. Chunks are committed lazily and
/// in order. Tracks occupancy and GC mark bits per slot, and
/// write-barrier dirty state per slot and per handle-word.
pub struct Page {
    /// Slot bytes for each committed chunk, `None` for uncommitted ones.
    /// Each chunk holds `PAGE_ENTRY_COUNT` slots of `slot_size` bytes,
    /// contiguously. Freed slots keep their bytes until reallocated;
    /// allocation always overwrites them.
    c_data: [Option<Box<[u8]>>; PAGE_ENTRY_COUNT_SZ],

    /// Bitmaps of occupied slots for each chunk. A slot is occupied from
    /// allocation until the sweep that frees it, whether or not it is
    /// still reachable.
    c_occupied: [BitMap64; PAGE_ENTRY_COUNT_SZ],

    /// Bitmaps of mark bits for each chunk. An occupied slot is live when
    /// its bit equals the page's `live_mark()`. Bits of unoccupied slots
    /// are stale and meaningless; allocation always writes the bit.
    c_mark: [BitMap64; PAGE_ENTRY_COUNT_SZ],

    /// Bitmaps of slots with at least one dirty handle-word, for each
    /// chunk. Only set while marking is in progress, and only on
    /// occupied slots. Set and cleared together with `h_dirty` and
    /// `s_dirty`.
    c_dirty: [BitMap64; PAGE_ENTRY_COUNT_SZ],

    /// Dirty bits for each committed chunk, `None` for uncommitted ones.
    /// One bit per handle-word across the chunk's contiguous bytes. Set
    /// and cleared together with `c_dirty` and `s_dirty`.
    h_dirty: [Option<Box<[BitMap64]>>; PAGE_ENTRY_COUNT_SZ],

    /// Summary of chunks that are fully occupied. Uncommitted chunks read
    /// as not full, so the lowest non-full chunk may need committing.
    s_occupied: BitMap64,

    /// Summary of chunks containing at least one dirty slot.
    s_dirty: BitMap64,

    /// The number of committed chunks. Chunks are always committed in
    /// order, so chunks `0..chunk_capacity` are committed.
    chunk_capacity: u32,

    /// The number of occupied slots, reachable or not, until a sweep frees
    /// the dead ones. Always equals the total population of `c_occupied`.
    slots_occupied: u32,

    /// The number of occupied slots whose mark bit equals `live_mark()`:
    /// objects marked live, or allocated black, since the page was last
    /// swept. Reset by each sweep, at which point it must equal the
    /// number of surviving slots.
    marked_live: u32,

    /// The index of the last sweep applied to this page. Its parity
    /// determines `live_mark()`. May lag behind the GC's sweep target;
    /// marking and allocation catch the page up before touching mark
    /// bits.
    sweep_index: u64,
    
    /// The size of each slot in bytes. A whole number of handle-words.
    slot_size: u32,

    /// Handle-words per slot: `slot_size / Handle::BYTES`. The dirty bit
    /// for a field is at `slot * slot_hw + field_hw` within the chunk.
    slot_hw: u32,
}

impl Page {

    /// Creates an empty page with no chunks committed.
    ///
    /// `sweep_index` should be the GC's current sweep target, so that the
    /// first allocation needs no catch-up. An older index is harmless,
    /// since an empty page catches up trivially on first use. A newer
    /// index will panic on first use.
    ///
    /// # Panics
    ///
    /// If `slot_size_class` is greater than 23 (64 KiB slots).
    /// 
    #[must_use]
    pub fn new(slot_size_class: u32, sweep_index: u64) -> Self {
        assert!(
            slot_size_class < CLASS_COUNT,
            "slot_size_class must be smaller than CLASS_COUNT"
        );
        let slot_size = class_bytes(slot_size_class);

        Self {
            c_data:     [const { None }; PAGE_ENTRY_COUNT_SZ],
            c_occupied: [BitMap64::NONE; PAGE_ENTRY_COUNT_SZ],
            c_mark:     [BitMap64::NONE; PAGE_ENTRY_COUNT_SZ],
            c_dirty:    [BitMap64::NONE; PAGE_ENTRY_COUNT_SZ],

            h_dirty: [const { None }; PAGE_ENTRY_COUNT_SZ],

            s_occupied: BitMap64::NONE,
            s_dirty:    BitMap64::NONE,
            marked_live: 0,
            sweep_index,

            chunk_capacity: 0,
            slots_occupied: 0,
            slot_size,
            slot_hw: slot_size / Handle::BYTES as u32,
        }
    }

}

impl Page {

    /// The number of occupied slots. On a page with a sweep pending, this
    /// includes dead objects until the page is swept up to date.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.slots_occupied as usize
    }

    /// The number of slots across all committed chunks, occupied or not.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        (self.chunk_capacity << PAGE_ENTRY_BITS) as usize
    }

    /// Whether no slots are occupied. A page with a sweep pending may
    /// report false while holding only dead objects.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.slots_occupied == 0
    }

    /// Whether every slot in every possible chunk is occupied, so no
    /// further allocation can succeed without a sweep.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.s_occupied.all()
    }

    /// The bytes committed for slot data across all chunks, excluding
    /// metadata.
    #[must_use]
    pub const fn bytes_total(&self) -> usize {
        self.capacity() * (self.slot_size as usize)
    }

    /// The bytes held by occupied slots. On a page with a sweep pending,
    /// this includes dead objects until the page is swept up to date.
    #[must_use]
    pub const fn bytes_used(&self) -> usize {
        self.len() * (self.slot_size as usize)
    }

}

impl Page {
    /// Claims the lowest free slot in the page, committing a new chunk if
    /// needed, and initializes it.
    ///
    /// The page is first caught up to the GC's sweep target. This may
    /// free dead slots, or reset an entirely dead page; any such sweep is
    /// reported in `AllocInfo::swept`.
    ///
    /// The slot is coloured by phase. During marking it is allocated
    /// black: marked live, guaranteed to survive the pending sweep, and
    /// initialized through the write barrier. Otherwise it is allocated
    /// white and will be collected by the next cycle if unreachable.
    ///
    /// `bytes` are written at offset 0 and any tail is zeroed; `None`
    /// zeroes the whole slot. Handle-words therefore never retain values
    /// from a previous occupant.
    /// 
    /// `handles` describes which handle-words of the slot's layout hold
    /// handles. It is only consulted when allocating black, to apply the
    /// write barrier to the initial contents.
    /// 
    /// # Errors
    ///
    /// `AllocError::PageFull` if every slot is occupied after catching up
    /// and no further chunks can be committed.
    ///
    /// # Panics
    ///
    /// If `bytes` is larger than the slot size, or if the GC's sweep
    /// target is behind the page's sweep index.
    /// 
    #[expect(
        clippy::panic_in_result_fn, 
        reason = "Calling the function with too many bytes is a logic error, not a runtime error."
    )]
    pub fn alloc(
        &mut self, 
        bytes: Option<&[u8]>,
        handles: HandleMap<'_>,
        gc: GcPhaseInfo
    ) -> Result<AllocInfo, AllocError> {

        assert!(bytes.map_or(0, <[u8]>::len) <= self.slot_size as usize, "Initial bytes exceed slot size");
        let swept = self.sweep_to(gc.sweep_target());

        // Find chunk with space, lowest first
        let chunk_idx = self.s_occupied.trailing_ones();
        if chunk_idx >= PAGE_ENTRY_COUNT {
            debug_assert!(swept.is_none_or(|s| s.freed == 0));
            return Err(AllocError::PageFull);
        }

        let black = gc.marking();

        // Allocate the first free slot, update bitmap states
        let slot_idx = self.alloc_slot_in(chunk_idx, black);
        let location = PageLocation { chunk: chunk_idx, slot: slot_idx };

        // Initialize the slot, through the barrier if allocated black
        let was_dirtied = self.write_all(location, bytes, handles, gc);

        // Result
        Ok(AllocInfo { location, was_dirtied, became_full: self.is_full(), swept })
    }

}

// // Immutable Access // //

impl Page {

    /// Whether the slot at `location` is occupied. Out-of-range locations
    /// and uncommitted chunks report false; this never panics.
    ///
    /// Occupied does not mean reachable: on a page with a sweep pending,
    /// dead objects remain occupied until swept.
    /// 
    #[must_use]
    pub const fn is_occupied(&self, location: PageLocation) -> bool {
        location.chunk < PAGE_ENTRY_COUNT
            && location.slot < PAGE_ENTRY_COUNT
            && self.c_occupied[location.chunk as usize].get(location.slot)
    }

    /// The bytes of the slot at `location`, or `None` if it is not
    /// occupied. Mutate through `write`, `write_all` or `write_handle` so
    /// the write barrier is applied.
    /// 
    #[must_use]
    pub fn get(&self, location: PageLocation) -> Option<&[u8]> {
        if self.is_occupied(location) {
            self.c_data[location.chunk as usize].as_deref()?.get(self.slot_range(location.slot))
        } else {
            None
        }
    }

}

impl Page {
    /// Writes a handle at the given handle-word offset within the slot.
    /// `hw_offset` must be a handle-word in the slot's layout.
    ///
    /// During marking, non-null handles written  are marked dirty so
    /// the marker will trace them. Outside marking nothing is dirtied;
    /// a reachable object will be found by the next marking cycle.
    /// Unlike `alloc`, a write never sweeps the page: the slot must
    /// already be occupied, and a reachable slot is never on a page
    /// the GC has abandoned.
    ///
    /// Returns true if the page newly became dirty, meaning it had no
    /// dirty marks before this write.
    ///
    /// # Panics
    ///
    /// If the slot is not occupied, or `hw_offset` lies outside the slot.
    /// 
    #[must_use]
    pub fn write_handle(&mut self, 
        location: PageLocation,
        hw_offset: u32,
        handle: Handle,
        gc: GcPhaseInfo
    ) -> bool {
        assert!(hw_offset < self.slot_hw, "handle offset lies outside the slot");
        assert!(self.is_occupied(location), "location not occupied");
        let at = (hw_offset << HANDLE_ALIGNMENT_BITS) as usize;
        self.at_mut_raw(location)[at..at + Handle::BYTES].copy_from_slice(&handle.to_ne_bytes());
        !handle.is_none() && gc.marking() && self.mark_dirty_raw(location, hw_offset)
    }

    /// Replaces the contents of a slot with `bytes`, zeroing any tail.
    /// `None` zeroes the slot entirely.
    ///
    /// During marking, non-null handles written  are marked dirty so
    /// the marker will trace them. Outside marking nothing is dirtied;
    /// a reachable object will be found by the next marking cycle.
    /// Unlike `alloc`, a write never sweeps the page: the slot must
    /// already be occupied, and a reachable slot is never on a page
    /// the GC has abandoned.
    ///
    /// Handles must be written whole, so `bytes` must not end partway
    /// through a handle-word; this is checked in debug builds.
    ///
    /// Returns true if the page newly became dirty, meaning it had no
    /// dirty marks before this write.
    ///
    /// # Panics
    ///
    /// If the slot is not occupied, or `bytes` is larger than the slot.
    /// 
    #[must_use]
    pub fn write_all(
        &mut self,
        location: PageLocation,
        bytes: Option<&[u8]>,
        handles: HandleMap<'_>,
        gc: GcPhaseInfo
    ) -> bool {
        assert!(self.is_occupied(location), "location not occupied");

        let slot_size_sz = self.slot_size as usize;
        let slot = self.at_mut_raw(location);
        let bytes_count = if let Some(bytes) = bytes {
            assert!(bytes.len() <= slot_size_sz, "bytes exceed slot size");
            debug_assert_no_torn_handles(handles, 0, bytes.len());

            slot[..bytes.len()].copy_from_slice(bytes);
            slot[bytes.len()..].fill(0);
            bytes.len()
        } else {
            slot.fill(0);
            0
        };

        gc.marking() && self.write_barrier_mark_region_dirty(
            location,
            0, 
            bytes_count, 
            handles
        )
    }

    /// Writes `bytes` into part of a slot, starting at byte offset `base`.
    ///
    /// During marking, non-null handles written  are marked dirty so
    /// the marker will trace them. Outside marking nothing is dirtied;
    /// a reachable object will be found by the next marking cycle.
    /// Unlike `alloc`, a write never sweeps the page: the slot must
    /// already be occupied, and a reachable slot is never on a page
    /// the GC has abandoned.
    ///
    /// Handles must be written whole at both ends of the range; torn
    /// writes are checked in debug builds.
    ///
    /// Returns true if the page newly became dirty, meaning it had no
    /// dirty marks before this write.
    ///
    /// # Panics
    ///
    /// If the slot is not occupied, or the write extends past the slot.
    /// 
    #[must_use]
    pub fn write(
        &mut self, 
        location: PageLocation,
        base: usize,
        bytes: &[u8],
        handles: HandleMap<'_>,
        gc: GcPhaseInfo
    ) -> bool {
        assert!(self.is_occupied(location), "location not occupied");

        assert!(
            base + bytes.len() <= (self.slot_size as usize), 
            "write extends past the end of the slot"
        );
        debug_assert_no_torn_handles(handles, base, bytes.len());

        let slot = self.at_mut_raw(location);
        slot[base..base+bytes.len()].copy_from_slice(bytes);

        gc.marking() && self.write_barrier_mark_region_dirty(
            location,  
            base, 
            bytes.len(), 
            handles
        )
    }

}

// // GC Mark // //

impl Page {
    /// Marks the slot at `location` live for the current marking phase,
    /// first catching the page up if it has a sweep pending.
    ///
    /// Returns whether the mark bit changed, along with any catch-up sweep
    /// that ran. The caller must apply the sweep to its accounting. Only
    /// newly marked objects need their handles traced. An object already
    /// marked has been queued for tracing, or, if allocated black, is
    /// covered by the write barrier.
    ///
    /// Must only be called while marking is in progress.
    ///
    /// # Panics
    ///
    /// If the page is more than one sweep behind the target. Every live
    /// object was either marked by the previous marking or allocated since,
    /// and both catch its page up to at least `target - 1`, so a larger gap
    /// means a stale handle is being traced. Also panics if the slot is
    /// unoccupied after catching up, for the same reason. In debug builds,
    /// also panics if called outside the marking phase.
    ///
    #[must_use]
    pub fn mark_live(&mut self, location: PageLocation, gc: GcPhaseInfo) -> (bool, Option<SweepInfo>) {
        debug_assert!(gc.marking(), "Marking outside the marking phase");
        let sweep_target = gc.sweep_target();
        assert!(
            sweep_target.saturating_sub(self.sweep_index) <= 1,
            "Page untouched by previous marking phase. Contents are dead, likely tracing stale handle."
        );
        let sweep_info = self.sweep_to(sweep_target);

        // Assert after sweep to catch marking dead location
        assert!(self.is_occupied(location), "Marking unoccupied slot, likely tracing a stale handle");
        let marked = self.c_mark[location.chunk as usize].set(location.slot, self.live_mark());
        self.marked_live += u32::from(marked);
        (marked, sweep_info)
    }

}

// // GC Dirty // //

impl Page {
    /// Marks a handle-word dirty without reading it, so the marker will
    /// trace its value when the page is drained. Should only be called
    /// while marking is in progress.
    ///
    /// `field_hw_offset` must lie within the slot, and the word must hold
    /// a handle. Null handles are skipped when the page is drained.
    ///
    /// Returns true if the page newly became dirty.
    ///
    /// # Panics
    ///
    /// If the slot is not occupied, or `field_hw_offset` lies outside the
    /// slot.
    /// 
    #[must_use]
    pub fn mark_dirty(&mut self, location: PageLocation, field_hw_offset: u32) -> bool {
        assert!(field_hw_offset < self.slot_hw, "handle offset lies outside the slot");
        assert!(self.is_occupied(location), "location not occupied");
        self.mark_dirty_raw(location, field_hw_offset)
    }

    /// Drains every dirty handle-word in the page, passing the current
    /// non-null value of each to `op`, then clears all dirty marks. The
    /// page is clean afterwards.
    ///
    /// Values are read at drain time, so a handle overwritten since it was
    /// marked dirty is passed with its newest value, or skipped if now
    /// null. This is the insertion-barrier contract: only current edges
    /// need tracing.
    ///
    /// Dirty marks only exist while marking is in progress. `op` cannot
    /// reference the page, so it should push into its own worklist. All
    /// dirty pages must be drained before the marking phase's atomic step.
    /// 
    /// # Panics
    ///
    /// Only if the page's own invariants are broken: a chunk flagged dirty
    /// in the summary that has no committed data or dirty map. That cannot
    /// result from any sequence of calls through the public API, so a
    /// panic here is a bug in the page, not in the caller.
    /// 
    pub fn process_dirty(&mut self, mut op: impl FnMut(Handle)) {
        let mut chunks = self.s_dirty;
        let mut chunk_idx = chunks.consume_zeros();
        while chunk_idx < PAGE_ENTRY_COUNT {
            let chunk_run = chunks.consume_ones();

            debug_assert!(chunk_idx + chunk_run <= self.chunk_capacity);
            for _ in 0..chunk_run {
                let chunk_idx_sz = chunk_idx as usize;

                let entries = self.h_dirty[chunk_idx_sz].as_mut().unwrap();
                let bytes = self.c_data[chunk_idx_sz].as_ref().unwrap();

                let mut slots = self.c_dirty[chunk_idx_sz];
                let mut slot_idx = slots.consume_zeros();
                while slot_idx < PAGE_ENTRY_COUNT {
                    let slot_run = slots.consume_ones();
                    let hw_begin = slot_idx * self.slot_hw;
                    let hw_end = (slot_idx + slot_run) * self.slot_hw;

                    for entry in (hw_begin >> PAGE_ENTRY_BITS)..=((hw_end - 1) >> PAGE_ENTRY_BITS) {
                        let mut bits = core::mem::take(&mut entries[entry as usize]).into_inner();
                        while bits != 0 {
                            let hw = (entry << PAGE_ENTRY_BITS) | bits.trailing_zeros();
                            bits &= bits - 1;
                            let at = (hw << HANDLE_ALIGNMENT_BITS) as usize;

                            let handle = Handle::from_ne_bytes(bytes[at..at + Handle::BYTES].try_into().unwrap());
                            if !handle.is_none() {
                                op(handle);
                            }
                        }
                    }

                    slot_idx += slot_run;
                    slot_idx += slots.consume_zeros();
                }
                self.c_dirty[chunk_idx_sz] = BitMap64::NONE;
                chunk_idx += 1;
            }
            chunk_idx += chunks.consume_zeros();
        }
        self.s_dirty = BitMap64::NONE;
    }

}

// // GC Sweep // //

/// The result of a sweep or reset applied to a page: slots freed and
/// any full or empty transitions. The caller must apply it to its
/// accounting.
/// 
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct SweepInfo {
    /// The number of slots freed.
    pub freed: u32,
    /// The page was full beforehand and now has at least one free slot.
    pub became_not_full: bool,
    /// The page held objects beforehand and now holds none.
    pub became_empty: bool,
}

impl Page {

    /// Brings the page's sweep index up to `target`, freeing whatever the
    /// marking for that sweep did not reach.
    ///
    /// A gap of zero does nothing. A gap of one runs a normal sweep,
    /// freeing every occupied slot left unmarked by the marking phase that
    /// fed `target`. A larger gap means the most recently completed
    /// marking reached none of the page's objects, since reaching any of
    /// them would have caught the page up to `target - 1`. Every object is
    /// therefore dead, and the page is reset instead.
    ///
    /// This is the entry point for background sweeping as well as for the
    /// page's own lazy catch-up. `target` must be a sweep whose marking
    /// has completed, normally `GcPhaseInfo::sweep_target`.
    ///
    /// Returns the sweep or reset performed, or `None` if the page was
    /// already up to date. The caller must apply it to its accounting.
    ///
    /// # Panics
    ///
    /// If `target` is behind the page's sweep index, or if a dirty mark
    /// would be lost: on a freed slot during a sweep, or anywhere in the
    /// page during a reset. Either indicates a barrier or drain ordering
    /// bug.
    ///
    #[must_use]
    pub fn sweep_to(&mut self, target: u64) -> Option<SweepInfo> {
        let gap = target.checked_sub(self.sweep_index)
            .expect("Sweep target behind page's sweep index");
        match gap {
            0 => None,
            1 => Some(self.process_sweep(target)),
            _ => Some(self.reset(target)),
        }
    }

}


// // Internal // //

impl Page {

    /// Internal.
    ///
    /// Accesses the slot data without checking occupancy. The chunk must be
    /// committed; out-of-range indices panic through indexing rather than
    /// an explicit check.
    ///
    /// # Panics
    ///
    /// If the chunk is not committed, or the location is out of range.
    /// 
    #[must_use]
    fn at_mut_raw(&mut self, location: PageLocation) -> &mut [u8] {
        let range = self.slot_range(location.slot);
        &mut self.c_data[location.chunk as usize].as_deref_mut().unwrap()[range]
    }

    /// Internal.
    ///
    /// Commits chunks until `chunk_capacity` is at least `chunks`,
    /// allocating zeroed slot data and an empty handle dirty map for each
    /// new chunk. Chunks are committed in order, so all chunks below
    /// `chunks` are committed afterwards.
    ///
    /// # Panics
    ///
    /// If `chunks` exceeds `PAGE_ENTRY_COUNT`.
    /// 
    fn commit_chunks_to(&mut self, chunks: u32) {
        assert!(chunks <= PAGE_ENTRY_COUNT);
        if chunks <= self.chunk_capacity { return; }
        self.c_data[(self.chunk_capacity as usize)..(chunks as usize)]
            .fill_with(|| Some(
                vec![0; (self.slot_size as usize) << PAGE_ENTRY_BITS]
                    .into_boxed_slice()
            ));

        self.h_dirty[(self.chunk_capacity as usize)..(chunks as usize)]
            .fill_with(|| Some(
                vec![BitMap64::NONE; self.slot_hw as usize]
                    .into_boxed_slice()
            ));
        self.chunk_capacity = chunks;
    }

    /// Internal.
    ///
    /// Claims the lowest free slot in `chunk_idx`, committing the chunk if
    /// needed, and records its colour. Black slots receive `live_mark()`
    /// and count towards `marked_live`. White slots receive the opposite
    /// bit and are not counted.
    ///
    /// The page must already be caught up to the sweep target, and the
    /// chunk must have a free slot. Returns the claimed slot's index
    /// within the chunk.
    /// 
    #[must_use]
    fn alloc_slot_in(&mut self, chunk_idx: u32, black: bool) -> u32 {
        // Ensure that the chunk data is allocated
        self.commit_chunks_to(chunk_idx + 1);

        let occupancy = &mut self.c_occupied[chunk_idx as usize];
        let slot_idx = occupancy.trailing_ones();
        debug_assert!(slot_idx < PAGE_ENTRY_COUNT, "Corrupted occupancy map");

        occupancy.mark(slot_idx);
        self.s_occupied.set(chunk_idx, occupancy.all());

        let colour = if black { self.live_mark() } else { !self.live_mark() };
        self.c_mark[chunk_idx as usize].set(slot_idx, colour);
    
        self.slots_occupied += 1;
        self.marked_live    += u32::from(black);

        slot_idx
    }

    /// Internal.
    ///
    /// Reads the handle-word at `field_hw_offset` and marks it dirty only
    /// if it is non-null. The slot must be occupied, the offset must lie
    /// within the slot, and the word must hold a handle.
    ///
    /// Returns true if the page newly became dirty.
    /// 
    fn mark_dirty_if_set_raw(&mut self, location: PageLocation, field_hw_offset: u32) -> bool {
        !self.read_handle_at_raw(location, field_hw_offset).is_none()
            && self.mark_dirty_raw(location, field_hw_offset)
    }

    /// Internal.
    ///
    /// Marks the handle-word at `field_hw_offset` within the slot as
    /// dirty. The slot must be occupied, the offset must lie within the
    /// slot, and the word must hold a handle.
    ///
    /// Returns true only if the page newly became dirty, having no
    /// previous dirty marks in `s_dirty`.
    /// 
    /// # Panics
    ///
    /// If the slot's chunk is not committed, or the handle-word lies past
    /// the end of the chunk's dirty map. An offset outside the slot but
    /// still inside the chunk dirties a neighbouring slot's word instead
    /// of panicking, so callers must check the offset themselves.
    /// 
    fn mark_dirty_raw(&mut self, location: PageLocation, field_hw_offset: u32) -> bool {
        let chunk_idx_sz = location.chunk as usize;

        let dirty_hw      = self.slot_hw_base(location.slot) + field_hw_offset;
        let dirty_hw_map  = (dirty_hw >> PAGE_ENTRY_BITS) as usize;
        let dirty_hw_bidx = dirty_hw & PAGE_ENTRY_MASK;

        let was_dirty = self.s_dirty.any();
        self.h_dirty[chunk_idx_sz].as_mut().unwrap()[dirty_hw_map].mark(dirty_hw_bidx);
        self.c_dirty[chunk_idx_sz].mark(location.slot);
        self.s_dirty.mark(location.chunk);
        !was_dirty 
    }

    /// Internal.
    ///
    /// Reads the handle-word at `field_hw_offset` within the slot. The
    /// slot's chunk must be committed, the offset must lie within the
    /// slot, and the word must hold a handle.
    /// 
    /// # Panics
    ///
    /// If the chunk is not committed, or the handle-word lies past the
    /// end of the chunk's data. An offset outside the slot but still
    /// inside the chunk reads a neighbouring slot's word instead of
    /// panicking, so callers must check the offset themselves.
    /// 
    fn read_handle_at_raw(&self, location: PageLocation, field_hw_offset: u32) -> Handle {
        let at = self.slot_range(location.slot).start + ((field_hw_offset as usize) << HANDLE_ALIGNMENT_BITS);
        let data = self.c_data[location.chunk as usize].as_deref().unwrap();
        Handle::from_ne_bytes(data[at..at + Handle::BYTES].try_into().unwrap())
    }

    /// Internal.
    ///
    /// Applies the write barrier to the byte range `byte_base` to
    /// `byte_base + byte_count` of the slot, marking every non-null handle
    /// in that range dirty. `handle_map` describes which handle-words of
    /// the slot hold handles.
    ///
    /// Returns true if the page newly became dirty.
    /// 
    #[must_use]
    fn write_barrier_mark_region_dirty(
        &mut self, 
        location: PageLocation,
        byte_base: usize,
        byte_count: usize,
        handle_map: HandleMap<'_>
    ) -> bool {
        if byte_count == 0 { return false; }
        let start_hw = (byte_base >> HANDLE_ALIGNMENT_BITS) as u32;
        let end_hw   = (byte_base + byte_count).div_ceil(Handle::BYTES) as u32;

        match handle_map {
            HandleMap::Scan(handle_count) => {
                self.write_barrier_mark_region_dirty_scan(
                    location,
                    start_hw,
                    end_hw,
                    handle_count
                )
            },
            HandleMap::Map(handle_bitmaps) => {
                self.write_barrier_mark_region_dirty_map(
                    location,
                    start_hw,
                    end_hw,
                    handle_bitmaps,
                )
            }
        }
    }

    /// Internal.
    ///
    /// Write barrier for layouts whose handles are the first
    /// `handle_count` handle-words of the slot. Marks every non-null
    /// handle in the overlap of that prefix and `start_hw..end_hw` dirty.
    ///
    /// Layouts with handles elsewhere, such as inline nested records, must
    /// use `write_barrier_mark_region_dirty_map` instead.
    ///
    /// Returns true if the page newly became dirty.
    /// 
    #[must_use]
    fn write_barrier_mark_region_dirty_scan(
        &mut self, 
        location: PageLocation,
        start_hw:  u32,
        end_hw:    u32,
        handle_count: u32,
    ) -> bool {
        let mut was_dirtied = false;
        for i in start_hw..end_hw.min(handle_count) {
            was_dirtied |= self.mark_dirty_if_set_raw(location, i);
        }
        was_dirtied
    }

    /// Internal.
    ///
    /// Write barrier for layouts with handles at arbitrary positions,
    /// described by one bit per handle-word in `handle_bitmaps`. Marks
    /// every non-null handle within `start_hw..end_hw` dirty.
    ///
    /// Layouts whose handles form a prefix of the slot should prefer
    /// `write_barrier_mark_region_dirty_scan`.
    ///
    /// Returns true if the page newly became dirty.
    /// 
    #[must_use]
    fn write_barrier_mark_region_dirty_map(
        &mut self, 
        location: PageLocation,
        start_hw:  u32,
        end_hw:    u32,
        handle_bitmaps: &[BitMap64],
    ) -> bool {
        let mut was_dirtied = false;
        let map_start = (start_hw >> PAGE_ENTRY_BITS) as usize;
        let map_end   = ((end_hw - 1) >> PAGE_ENTRY_BITS) as usize;
        for (idx, &word) in handle_bitmaps.iter().enumerate().take(map_end+1).skip(map_start) {
            let span_start = if idx == map_start { 
                start_hw & PAGE_ENTRY_MASK 
            } else { 
                0 
            };

            let span_end = if idx == map_end {
                ((end_hw - 1) & PAGE_ENTRY_MASK) + 1 
            } else { 
                PAGE_ENTRY_COUNT
            };

            let mut bits = (word & BitMap64::span(span_start, span_end-span_start)).into_inner();
            while bits != 0 {
                let hw = ((idx as u32) << PAGE_ENTRY_BITS) | bits.trailing_zeros();
                bits &= bits - 1;
                was_dirtied |= self.mark_dirty_if_set_raw(location, hw);
            }
        }
        was_dirtied
    }

    /// Internal.
    ///
    /// The mark-bit value meaning "live" for the marking that feeds this
    /// page's next sweep: the parity of `sweep_index`. It flips each time
    /// the page is swept, so survivors become unmarked without clearing.
    /// 
    const fn live_mark(&self) -> bool {
        self.sweep_index & 0x1 == 0x1
    }

    /// Internal.
    ///
    /// Advances the page by exactly one sweep, freeing every occupied slot
    /// whose mark bit differs from the page's current `live_mark()`.
    /// Survivors keep their mark bits, which read as unmarked under the
    /// new index's parity, so no clearing pass is needed.
    ///
    /// `sweep_index` must be the page's index plus one, and the marking
    /// phase feeding it must have completed. Use `sweep_to` for gaps
    /// larger than one.
    ///
    /// Returns the number of slots freed and the resulting full and empty
    /// transitions.
    ///
    /// # Panics
    ///
    /// If `sweep_index` is not the next sequential index, or if a freed
    /// slot has a dirty mark, which indicates a barrier or drain ordering
    /// bug.
    /// 
    fn process_sweep(&mut self, sweep_index: u64) -> SweepInfo {
        assert_eq!(self.sweep_index + 1, sweep_index, "Expected next sequential sweep");
        let live_mark = self.live_mark();
        
        let was_full  = self.is_full();
        let was_empty = self.is_empty();

        // XOR mask turning raw mark bits into a live map for this page's parity
        let flip = if live_mark { BitMap64::NONE } else { BitMap64::ALL };

        // Assert no dirty marks on dead slots
        for chunk_idx in 0..self.chunk_capacity {
            let chunk_idx_sz = chunk_idx as usize;
            let live  = self.c_mark[chunk_idx_sz] ^ flip;
            let freed = self.c_occupied[chunk_idx_sz] & !live;
            assert!(
                (self.c_dirty[chunk_idx_sz] & freed).none(),
                "Expected no dirty marks on dead slot"
            );
        }

        // Collect garbage
        let mut garbage  = 0;
        let mut occupied = 0;
        for chunk_idx in 0..self.chunk_capacity {
            let chunk_idx_sz = chunk_idx as usize;
            let live  = self.c_mark[chunk_idx_sz] ^ flip;
            let freed = self.c_occupied[chunk_idx_sz] & !live;

            // Free dead slots
            self.c_occupied[chunk_idx_sz] &= live;
            garbage += freed.count_ones();

            // Update summaries
            self.s_occupied.set(chunk_idx, self.c_occupied[chunk_idx_sz].all());
            occupied += self.c_occupied[chunk_idx_sz].count_ones();
        }

        debug_assert_eq!(occupied,           self.marked_live   );
        debug_assert_eq!(occupied + garbage, self.slots_occupied);
        self.slots_occupied = occupied;
        self.marked_live = 0;
        self.sweep_index = sweep_index;

        SweepInfo {
            freed: garbage,
            became_not_full: was_full && self.s_occupied.nall(),
            became_empty: !was_empty && self.is_empty(),
        }
    }

    /// Internal.
    ///
    /// Frees every slot and sets the page's sweep index to `sweep_index`.
    /// Used for pages that no marking has touched for at least one full
    /// cycle. Chunks stay committed, and stale mark bits are left in
    /// place, since allocation always writes the bit.
    ///
    /// # Panics
    ///
    /// If any chunk has a dirty mark. An entirely dead page cannot have
    /// been written through the barrier.
    /// 
    fn reset(&mut self, sweep_index: u64) -> SweepInfo {
        let was_full  = self.is_full();
        let was_empty = self.is_empty();
        let freed     = self.slots_occupied;

        assert!(self.s_dirty.none(), "Dirty mark in dead page");
        for chunk in 0..self.chunk_capacity as usize {
            assert!(self.c_dirty[chunk].none(), "Dirty mark in dead page");
            self.c_occupied[chunk] = BitMap64::NONE;
        }
        self.s_occupied     = BitMap64::NONE;
        self.slots_occupied = 0;
        self.marked_live    = 0;
        self.sweep_index    = sweep_index;

        SweepInfo { freed, became_not_full: was_full, became_empty: !was_empty }
    }

}

impl Page {
    /// The byte range of `slot_idx` within its chunk's data.
    #[inline]
    const fn slot_range(&self, slot_idx: u32) -> Range<usize> {
        let from = (slot_idx * self.slot_size) as usize;
        from..from + self.slot_size as usize
    }

    /// The index of the slot's first handle-word within its chunk.
    const fn slot_hw_base(&self, slot_idx: u32) -> u32 {
        slot_idx * self.slot_hw
    }
}

/// Asserts, in debug builds, that a write of `len` bytes at `base`
/// neither starts nor ends partway through a handle-word described by
/// `handles`.
/// 
fn debug_assert_no_torn_handles(handles: HandleMap<'_>, base: usize, len: usize) {
    if len == 0 { return; }
    let end = base + len;
    let unaligned = |at: usize| at & (Handle::BYTES - 1) != 0;

    let first_hw = (base >> HANDLE_ALIGNMENT_BITS) as u32;
    debug_assert!(
        !(unaligned(base) && handles.is_handle(first_hw)),
        "write at byte {base} starts inside handle word {first_hw}",
    );

    let last_hw = ((end - 1) >> HANDLE_ALIGNMENT_BITS) as u32;
    debug_assert!(
        !(unaligned(end) && handles.is_handle(last_hw)),
        "write ending at byte {end} stops inside handle word {last_hw}",
    );
}


#[cfg(test)]
mod tests;