// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

//! Unit tests for `Page`. Every test checks `Page::check_invariants`
//! after each step, so individual tests focus on behaviour rather than
//! accounting.
//!
//! Pages are built by class index. Class 0 is 16 B, the smallest, and
//! packs 32 slots into each dirty word; class 2 is 48 B, six
//! handle-words, so its slots straddle dirty words; class 23 is 64 KiB,
//! the largest.
//! 
//! **Disclosure** Unless otherwise stated on the test, these tests
//!     were written by an LLM and were not thoroughly reviewed by a
//!     human.

#![allow(
    clippy::missing_panics_doc,
    reason = "tests and their helpers panic by design; the assertion messages are the documentation"
)]

use alloc::{collections::{BTreeMap, BTreeSet}, vec::Vec};

use super::*;
use crate::heap::class_for;

/// Slots per page.
const PAGE_SLOTS: usize = PAGE_ENTRY_COUNT_SZ * PAGE_ENTRY_COUNT_SZ;

impl Page {
    /// Asserts every cross-structure invariant of the page: the counters
    /// match the bitmaps, the summaries match the chunk maps, and the
    /// three dirty levels agree with each other and with occupancy.
    ///
    /// Checks that chunks below `chunk_capacity` are committed at the
    /// right sizes and those above hold no data, dirty map, occupancy or
    /// marks, that `slots_occupied` and `marked_live` match the bitmaps,
    /// that `s_occupied` and `s_dirty` match the chunk maps, that every
    /// `h_dirty` bit lies in a `c_dirty` slot and every `c_dirty` slot has
    /// an `h_dirty` bit, and that `c_dirty` is a subset of `c_occupied`.
    ///
    fn check_invariants(&self) {
        let chunk_capacity_sz = self.chunk_capacity as usize;

        // // Check Basics // //

        assert!(self.chunk_capacity <= PAGE_ENTRY_COUNT, "chunk_capacity exceeds the page");
        assert_eq!(
            self.slot_hw as usize * Handle::BYTES, self.slot_size as usize,
            "slot_hw disagrees with slot_size",
        );

        for i in 0..chunk_capacity_sz {
            assert_eq!(
                self.c_data[i].as_deref().map(<[u8]>::len),
                Some(self.slot_size as usize * PAGE_ENTRY_COUNT_SZ),
                "c_data missing or wrongly sized at committed chunk {i}",
            );
            assert_eq!(
                self.h_dirty[i].as_deref().map(<[BitMap64]>::len),
                Some(self.slot_hw as usize),
                "h_dirty missing or wrongly sized at committed chunk {i}",
            );
        }

        for i in chunk_capacity_sz..PAGE_ENTRY_COUNT_SZ {
            assert!(self.c_data[i].is_none(),  "c_data set in uncommitted chunk {i}");
            assert!(self.h_dirty[i].is_none(), "h_dirty set in uncommitted chunk {i}");
            assert!(self.c_occupied[i].none(), "c_occupied set in uncommitted chunk {i}");
            assert!(self.c_mark[i].none(),     "c_mark set in uncommitted chunk {i}");
        }

        // // Check Page Summaries // //

        let s_occupied_calc = BitMap64::summarize_all(BitMap64::NONE, 0, &self.c_occupied);
        assert_eq!(s_occupied_calc, self.s_occupied, "s_occupied disagrees with c_occupied");

        let s_dirty_calc = BitMap64::summarize_any(BitMap64::NONE, 0, &self.c_dirty);
        assert_eq!(s_dirty_calc, self.s_dirty, "s_dirty disagrees with c_dirty");

        // // Check Page Totals // //

        let c_occupied_total: u32 = self.c_occupied.iter()
            .copied()
            .map(BitMap64::count_ones)
            .sum();
        assert_eq!(c_occupied_total, self.slots_occupied, "slots_occupied disagrees with c_occupied_total");

        let c_mark_live: u32 = self.c_mark.iter().zip(self.c_occupied.iter())
            .map(|(&marks, &occ)| if self.live_mark() {
                (marks & occ).count_ones()
            } else {
                (!marks & occ).count_ones()
            })
            .sum();
        assert_eq!(c_mark_live, self.marked_live, "c_mark disagrees with marked_live");

        // // Check Chunk Summaries // //

        for chunk in 0..PAGE_ENTRY_COUNT_SZ {
            assert!(
                (self.c_dirty[chunk] & !self.c_occupied[chunk]).none(),
                "c_dirty set on unoccupied slot in chunk {chunk}",
            );

            match self.h_dirty[chunk].as_deref() {
                Some(h_dirty) => assert_eq!(
                    self.c_dirty[chunk],
                    self.summarize_h_dirty(h_dirty),
                    "c_dirty disagrees with h_dirty in chunk {chunk}",
                ),
                None => assert!(
                    self.c_dirty[chunk].none(),
                    "c_dirty set in uncommitted chunk {chunk}",
                ),
            }
        }
    }

    /// Rebuilds a chunk's slot-dirty bitmap from its handle-word dirty
    /// bits. A slot is dirty when any handle-word in its range is dirty.
    ///
    /// Walks every handle-word of every slot individually, so it stays
    /// correct when a slot's handle-words straddle two dirty words.
    ///
    fn summarize_h_dirty(&self, h_dirty: &[BitMap64]) -> BitMap64 {
        let mut summary = BitMap64::NONE;
        for slot in 0..PAGE_ENTRY_COUNT {
            let base = self.slot_hw_base(slot);
            let dirty = (base..base + self.slot_hw).any(|hw| {
                h_dirty[(hw >> PAGE_ENTRY_BITS) as usize].get(hw & PAGE_ENTRY_MASK)
            });
            summary.set(slot, dirty);
        }
        summary
    }
}

// // Helpers // //

/// A GC phase with marking in progress for `mark_epoch`.
fn marking(mark_epoch: u64) -> GcPhaseInfo {
    GcPhaseInfo::new(mark_epoch, true)
}

/// A GC phase with no marking in progress, at `mark_epoch`.
fn idle(mark_epoch: u64) -> GcPhaseInfo {
    GcPhaseInfo::new(mark_epoch, false)
}

/// A non-null handle of kind 1 carrying `n` in its low bits, so tests
/// can tell handles apart.
fn handle(n: u64) -> Handle {
    Handle((1 << 60) | n)
}

/// A kind-0 word that is not all zero: null, but not `NONE`.
const KIND_ZERO_WORD: Handle = Handle(0x0000_FFFF_FFFF_FFFF);

const fn loc(chunk: u32, slot: u32) -> PageLocation {
    PageLocation { chunk, slot }
}

/// Allocates `count` empty slots under `gc`, checking invariants after
/// each, and returns their locations in order.
fn alloc_n(p: &mut Page, count: usize, gc: GcPhaseInfo) -> Vec<PageLocation> {
    (0..count).map(|_| {
        let info = p.alloc(None, HandleMap::Scan(0), gc).expect("page has free slots");
        p.check_invariants();
        info.location
    }).collect()
}

/// Marks every location live under `gc`, asserting each was newly
/// marked, and checks invariants once at the end.
fn mark_all(p: &mut Page, locs: &[PageLocation], gc: GcPhaseInfo) {
    for &l in locs {
        let (marked, _) = p.mark_live(l, gc);
        assert!(marked, "{l:?} was already marked");
    }
    p.check_invariants();
}

/// Drains the page, asserting it is clean afterwards, and returns the
/// raw handle words it reported in sorted order.
fn drain_sorted(p: &mut Page) -> Vec<u64> {
    let mut seen = Vec::new();
    p.process_dirty(|h| seen.push(h.0));
    p.check_invariants();
    assert!(p.s_dirty.none(), "page still dirty after drain");
    seen.sort_unstable();
    seen
}

/// The raw words of `handles` in sorted order, for comparing against
/// `drain_sorted`.
fn raw(handles: &[Handle]) -> Vec<u64> {
    let mut words: Vec<u64> = handles.iter().map(|h| h.0).collect();
    words.sort_unstable();
    words
}

/// A slot's worth of bytes holding `handle(n + 1)` in handle-word `n`.
fn handle_bytes(slot_hw: u32) -> Vec<u8> {
    (0..u64::from(slot_hw)).flat_map(|n| handle(n + 1).to_ne_bytes()).collect()
}

/// Builds a 32-byte-slot page whose slot 0 was filled with `0xFF` and
/// then freed by a sweep, so its stale bytes are still present.
///
fn page_with_stale_slot() -> Page {
    let mut p = Page::new(1, 0);
    let info = p.alloc(Some(&[0xFF; 32]), HandleMap::Scan(0), idle(0))
        .expect("page has free slots");
    assert_eq!(info.location, loc(0, 0));

    // A marking that reached nothing, so the sweep frees slot 0
    let swept = p.sweep_to(1);
    assert_eq!(swept, Some(SweepInfo { freed: 1, became_not_full: false, became_empty: true }));
    p.check_invariants();

    // Freed slots keep their bytes, so the stale data is still there
    let data = p.c_data[0].as_deref().expect("chunk 0 is committed");
    assert!(data[p.slot_range(0)].iter().all(|&byte| byte == 0xFF));

    p
}

mod classes {
    use super::*;

    /// The table starts at 16 B, steps by 16 B to 256 B, then doubles to
    /// 64 KiB, and every size is whole handle-words.
    #[test]
    fn table_shape() {
        assert_eq!(class_bytes(0), 16);
        assert_eq!(class_bytes(15), 256);
        assert_eq!(class_bytes(16), 512);
        assert_eq!(class_bytes(CLASS_COUNT - 1), 1 << 16);
        for class in 1..CLASS_COUNT {
            if class < 16 {
                assert_eq!(class_bytes(class) - class_bytes(class - 1), 16, "class {class}");
            } else {
                assert_eq!(class_bytes(class), 2 * class_bytes(class - 1), "class {class}");
            }
            assert!(class_bytes(class).is_multiple_of(Handle::BYTES as u32));
        }
    }

    /// `class_for` picks the smallest class that fits: a class's own size
    /// and one byte under map to itself, one byte over maps to the next.
    #[test]
    fn class_for_round_trips() {
        assert_eq!(class_for(0), 0);
        assert_eq!(class_for(1), 0);
        for class in 0..CLASS_COUNT {
            let bytes = class_bytes(class);
            assert_eq!(class_for(bytes), class, "exact size of class {class}");
            assert_eq!(class_for(bytes - 1), class, "one under class {class}");
            if class + 1 < CLASS_COUNT {
                assert_eq!(class_for(bytes + 1), class + 1, "one over class {class}");
            }
        }
    }    
    
    /// A class past the end of the table panics.
    #[test]
    #[should_panic(expected = "smaller than CLASS_COUNT")]
    fn new_rejects_class_past_table() {
        let _ = Page::new(CLASS_COUNT, 0);
    }
}

mod allocation {
    use super::*;

    /// Every class constructs, at even and odd sweep indices, with the
    /// handle-word count derived from the byte size.
    #[test]
    fn new_accepts_every_class() {
        for class in 0..CLASS_COUNT {
            for index in [0, 1, 7] {
                let p = Page::new(class, index);
                p.check_invariants();
                assert_eq!(p.slot_hw as usize * Handle::BYTES, class_bytes(class) as usize);
            }
        }
    }

    /// Allocations fill slot by slot, and the 65th commits chunk 1.
    #[test]
    fn fills_lowest_first() {
        let mut p = Page::new(0, 0);
        p.check_invariants();

        for i in 0..PAGE_ENTRY_COUNT {
            let result = p.alloc(None, HandleMap::Scan(0), idle(0))
                .expect("page has free slots");
            p.check_invariants();

            assert_eq!(result.location, loc(0, i));
            assert_eq!(p.capacity(), PAGE_ENTRY_COUNT_SZ);
            assert!(!result.became_full);
            assert!(!result.was_dirtied);
        }

        let result = p.alloc(None, HandleMap::Scan(0), idle(0))
            .expect("page has free slots");
        p.check_invariants();

        assert_eq!(result.location, loc(1, 0));
        assert_eq!(p.capacity(), 2 * PAGE_ENTRY_COUNT_SZ);
        assert!(!result.became_full);
        assert!(!result.was_dirtied);
    }

    /// After freeing slots 5 and 2, the next allocations take 2, then 5.
    #[test]
    fn reuses_lowest_freed_slot() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, 8, idle(0));
        assert_eq!(locs[7], loc(0, 7));

        // Mark everything but slot 2 and slot 5 as live
        for i in (0..8).filter(|&slot| slot != 2 && slot != 5) {
            let (marked, swept) = p.mark_live(loc(0, i), marking(1));
            p.check_invariants();
            assert!(marked);
            assert_eq!(swept, None);
        }

        // Moving to idle(1) triggers the sweep on the first alloc, and
        // the next two allocations take 2 and then 5
        let info = p.alloc(None, HandleMap::Scan(0), idle(1))
            .expect("page has free slots");
        p.check_invariants();
        assert_eq!(info.swept, Some(SweepInfo { freed: 2, became_not_full: false, became_empty: false }));
        assert_eq!(info.location, loc(0, 2));

        let info = p.alloc(None, HandleMap::Scan(0), idle(1))
            .expect("page has free slots");
        p.check_invariants();
        assert_eq!(info.swept, None);
        assert_eq!(info.location, loc(0, 5));

        // Then we continue from the tail
        let info = p.alloc(None, HandleMap::Scan(0), idle(1))
            .expect("page has free slots");
        p.check_invariants();
        assert_eq!(info.swept, None);
        assert_eq!(info.location, loc(0, 8));
    }

    /// `became_full` is set only on the last allocation, and the next
    /// allocation returns `PageFull`.
    #[test]
    fn fills_page_then_reports_full() {
        let mut p = Page::new(0, 0);
        p.check_invariants();

        for i in 0..PAGE_SLOTS - 1 {
            let info = p.alloc(None, HandleMap::Scan(0), idle(0))
                .expect("page has free slots");
            p.check_invariants();

            let slot  = (i as u32) & PAGE_ENTRY_MASK;
            let chunk = (i >> PAGE_ENTRY_BITS) as u32;

            assert_eq!(info.swept, None);
            assert_eq!(info.location, loc(chunk, slot));
            assert!(!info.was_dirtied);
            assert!(!info.became_full);
        }

        let info = p.alloc(None, HandleMap::Scan(0), idle(0))
            .expect("page has free slots");

        assert_eq!(info.swept, None);
        assert_eq!(info.location, loc(PAGE_ENTRY_COUNT - 1, PAGE_ENTRY_COUNT - 1));
        assert!(!info.was_dirtied);
        assert!(info.became_full);
        assert!(p.is_full());

        // A full page with nothing to sweep refuses further allocation
        let full = p.alloc(None, HandleMap::Scan(0), idle(0)).err();
        p.check_invariants();
        assert_eq!(full, Some(AllocError::PageFull));
    }

    /// Initial bytes larger than the slot panic.
    #[test]
    #[should_panic(expected = "Initial bytes exceed slot size")]
    fn oversized_bytes() {
        let mut p = Page::new(0, 0);
        let _ = p.alloc(Some(&[0; 17]), HandleMap::Scan(0), idle(0));
    }

    /// `Some(short)` into a stale freed slot leaves no stale tail bytes.
    #[test]
    fn realloc_zeroes_tail() {
        let mut p = page_with_stale_slot();

        let info = p.alloc(Some(&[1, 2, 3]), HandleMap::Scan(0), idle(1))
            .expect("page has free slots");
        p.check_invariants();
        assert_eq!(info.location, loc(0, 0));
        assert_eq!(info.swept, None);

        let mut expected = [0; 32];
        expected[..3].copy_from_slice(&[1, 2, 3]);
        assert_eq!(p.get(info.location), Some(&expected[..]));
    }

    /// `None` into a stale freed slot zeroes the whole slot.
    #[test]
    fn realloc_none_zeroes_slot() {
        let mut p = page_with_stale_slot();

        let info = p.alloc(None, HandleMap::Scan(0), idle(1))
            .expect("page has free slots");
        p.check_invariants();
        assert_eq!(info.location, loc(0, 0));
        assert_eq!(info.swept, None);

        assert_eq!(p.get(info.location), Some(&[0; 32][..]));
    }

    /// Allocation always writes the slot's colour, even over a stale mark
    /// bit that happens to read as meaningful. White over a stale bit that
    /// reads live must come out unmarked, and black over a stale bit that
    /// reads dead must come out marked.
    ///
    #[test]
    fn realloc_overwrites_stale_colour() {
        let location = loc(0, 0);

        // White over live: the sweep to 1 left slot 0's stale bit equal to
        // the new live mark, so a skipped colour write would allocate black.
        let mut p = page_with_stale_slot();
        let info = p.alloc(None, HandleMap::Scan(0), idle(1))
            .expect("page has free slots");
        p.check_invariants();
        assert_eq!(info.location, location);
        assert_eq!(info.swept, None);
        assert_eq!(p.marked_live, 0);

        let (marked, swept) = p.mark_live(location, marking(2));
        p.check_invariants();
        assert!(marked, "white allocation read as already marked");
        assert_eq!(swept, None);

        // Black over dead: one more sweep of the empty page flips the live
        // mark, so the stale bit now reads dead and a skipped colour write
        // would allocate white.
        let mut p = page_with_stale_slot();
        let swept = p.sweep_to(2);
        p.check_invariants();
        assert_eq!(swept, Some(SweepInfo::default()));

        let info = p.alloc(None, HandleMap::Scan(0), marking(3))
            .expect("page has free slots");
        p.check_invariants();
        assert_eq!(info.location, location);
        assert_eq!(info.swept, None);
        assert_eq!(p.marked_live, 1);

        let (marked, swept) = p.mark_live(location, marking(3));
        p.check_invariants();
        assert!(!marked, "black allocation read as unmarked");
        assert_eq!(swept, None);
    }

    /// A catch-up sweep that frees slots on a full page is reported in
    /// `swept` with `became_not_full`, and the allocation that triggered it
    /// then succeeds without reporting `became_full`. Chunks stay
    /// committed through the sweep.
    ///
    #[test]
    fn alloc_sweeps_full_page() {
        let mut p = Page::new(0, 0);
        let _ = alloc_n(&mut p, PAGE_SLOTS, idle(0));
        assert!(p.is_full());

        let info = p.alloc(None, HandleMap::Scan(0), idle(1))
            .expect("the catch-up sweep frees the page");
        p.check_invariants();
        assert_eq!(info.swept, Some(SweepInfo {
            freed: PAGE_SLOTS as u32,
            became_not_full: true,
            became_empty: true,
        }));
        assert_eq!(info.location, loc(0, 0));
        assert!(!info.became_full);
        assert_eq!(p.len(), 1);
        assert_eq!(p.capacity(), PAGE_SLOTS);
    }

    /// A page several sweeps behind is reset by its first allocation.
    #[test]
    fn alloc_resets_lagging_page() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, 3, idle(0));

        let info = p.alloc(None, HandleMap::Scan(0), idle(2))
            .expect("page has free slots");
        p.check_invariants();
        assert_eq!(info.swept, Some(SweepInfo { freed: 3, became_not_full: false, became_empty: true }));
        assert_eq!(info.location, locs[0]);
        assert_eq!(p.len(), 1);
        assert!(!p.is_occupied(locs[1]));
    }

    /// A black allocation whose initial bytes hold a non-null handle dirties
    /// the page. A second such allocation on the already-dirty page does not
    /// report it, a handle the layout does not describe is ignored, and a
    /// black allocation holding only null words never dirties.
    ///
    #[test]
    fn alloc_black_dirties() {
        let mut p = Page::new(0, 0);
        let a = p.alloc(Some(&handle(1).to_ne_bytes()), HandleMap::Scan(1), marking(1))
            .expect("page has free slots");
        p.check_invariants();
        assert!(a.was_dirtied);
        assert!(p.s_dirty.any());

        let b = p.alloc(Some(&handle(2).to_ne_bytes()), HandleMap::Scan(1), marking(1))
            .expect("page has free slots");
        p.check_invariants();
        assert!(!b.was_dirtied);
        assert_eq!(drain_sorted(&mut p), raw(&[handle(1), handle(2)]));

        // A handle outside the described layout is not a handle
        let mut p = Page::new(0, 0);
        let c = p.alloc(Some(&handle(3).to_ne_bytes()), HandleMap::Scan(0), marking(1))
            .expect("page has free slots");
        p.check_invariants();
        assert!(!c.was_dirtied);
        assert!(p.s_dirty.none());

        // Null words never dirty, whether written or implied
        let mut p = Page::new(0, 0);
        let d = p.alloc(Some(&Handle::NONE.to_ne_bytes()), HandleMap::Scan(1), marking(1))
            .expect("page has free slots");
        let e = p.alloc(Some(&KIND_ZERO_WORD.to_ne_bytes()), HandleMap::Scan(1), marking(1))
            .expect("page has free slots");
        let f = p.alloc(None, HandleMap::Scan(2), marking(1))
            .expect("page has free slots");
        p.check_invariants();
        assert!(!d.was_dirtied && !e.was_dirtied && !f.was_dirtied);
        assert!(p.s_dirty.none());
    }

    /// A white allocation never applies the barrier, even when its
    /// initial bytes hold handles.
    #[test]
    fn alloc_white_ignores_handles() {
        let mut p = Page::new(0, 0);
        let info = p.alloc(Some(&handle(1).to_ne_bytes()), HandleMap::Scan(1), idle(0))
            .expect("page has free slots");
        p.check_invariants();
        assert!(!info.was_dirtied);
        assert!(p.s_dirty.none());
        assert_eq!(p.marked_live, 0);
    }

    /// A page created with a sweep index ahead of the GC's sweep target
    /// panics on first use.
    #[test]
    #[should_panic(expected = "Sweep target behind page's sweep index")]
    fn new_ahead_of_target() {
        let mut p = Page::new(0, 5);
        let _ = p.alloc(None, HandleMap::Scan(0), idle(1));
    }

    /// `len`, `is_empty`, `bytes_used` and `bytes_total` follow allocation,
    /// chunk commits and sweeps. Chunks stay committed after a sweep, so
    /// `bytes_total` never drops.
    ///
    #[test]
    fn accounting_follows_alloc_and_sweep() {
        let mut p = Page::new(0, 0);
        assert_eq!((p.len(), p.capacity(), p.bytes_used(), p.bytes_total()), (0, 0, 0, 0));
        assert!(p.is_empty());
        assert!(!p.is_full());

        let locs = alloc_n(&mut p, 70, idle(0));
        assert_eq!(p.len(), 70);
        assert_eq!(p.capacity(), 128);
        assert_eq!(p.bytes_used(), 70 * 16);
        assert_eq!(p.bytes_total(), 128 * 16);
        assert!(!p.is_empty());

        mark_all(&mut p, &locs[..10], marking(1));
        let swept = p.sweep_to(1);
        p.check_invariants();
        assert_eq!(swept, Some(SweepInfo { freed: 60, became_not_full: false, became_empty: false }));
        assert_eq!(p.len(), 10);
        assert_eq!(p.bytes_used(), 10 * 16);
        assert_eq!(p.capacity(), 128);
        assert_eq!(p.bytes_total(), 128 * 16);

        // Nothing marked for sweep 2, so the survivors go too
        let swept = p.sweep_to(2);
        p.check_invariants();
        assert_eq!(swept, Some(SweepInfo { freed: 10, became_not_full: false, became_empty: true }));
        assert!(p.is_empty());
        assert_eq!(p.bytes_used(), 0);
        assert_eq!(p.bytes_total(), 128 * 16);
    }
}

mod access {
    use super::*;

    /// Out-of-range, uncommitted and freed locations read as unoccupied
    /// without panicking.
    #[test]
    fn unoccupied_reads_as_none() {
        let mut p = Page::new(0, 0);
        let _ = alloc_n(&mut p, 1, idle(0));

        let probes = [
            loc(0, 1), loc(1, 1), loc(0, PAGE_ENTRY_COUNT),
            loc(PAGE_ENTRY_COUNT, 1), loc(u32::MAX, u32::MAX),
        ];
        assert!(p.get(loc(0, 0)).is_some());
        for probe in probes { assert!(p.get(probe).is_none(), "{probe:?}"); }

        let swept = p.sweep_to(1);
        p.check_invariants();
        assert_eq!(swept, Some(SweepInfo { freed: 1, became_not_full: false, became_empty: true }));

        assert!(p.get(loc(0, 0)).is_none());
        for probe in probes { assert!(p.get(probe).is_none(), "{probe:?}"); }
    }

    /// A slot freed by a reset reads as unoccupied.
    #[test]
    fn reset_slot_reads_as_none() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, idle(0))[0];
        let _ = p.sweep_to(2);
        p.check_invariants();
        assert!(!p.is_occupied(l));
        assert!(p.get(l).is_none());
    }

    /// A partial `write` leaves the surrounding bytes untouched.
    #[test]
    fn write_preserves_surrounding_bytes() {
        let mut p = Page::new(0, 0);
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8];
        let location = p.alloc(Some(&bytes), HandleMap::Scan(0), idle(0))
            .expect("page has free slots")
            .location;
        p.check_invariants();
        assert!(!p.write(location, 1, &[4, 3, 2], HandleMap::Scan(0), idle(0)));
        assert_eq!(p.get(location).unwrap(), &[1, 4, 3, 2, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
        p.check_invariants();
    }

    /// `write_all` replaces the slot and zeroes any tail; `None` zeroes
    /// it entirely.
    #[test]
    fn write_all_zeroes_tail() {
        let mut p = Page::new(0, 0);
        let location = p.alloc(Some(&[0xAA; 16]), HandleMap::Scan(0), idle(0))
            .expect("page has free slots")
            .location;

        assert!(!p.write_all(location, Some(&[1, 2, 3]), HandleMap::Scan(0), idle(0)));
        p.check_invariants();
        let mut expected = [0; 16];
        expected[..3].copy_from_slice(&[1, 2, 3]);
        assert_eq!(p.get(location), Some(&expected[..]));

        assert!(!p.write_all(location, None, HandleMap::Scan(0), idle(0)));
        p.check_invariants();
        assert_eq!(p.get(location), Some(&[0; 16][..]));
    }

    /// `write_handle` stores the handle's bytes at the handle-word and
    /// touches nothing else.
    #[test]
    fn write_handle_round_trips() {
        let mut p = Page::new(0, 0);
        let location = p.alloc(Some(&[0xAA; 16]), HandleMap::Scan(2), idle(0))
            .expect("page has free slots")
            .location;

        assert!(!p.write_handle(location, 1, handle(9), idle(0)));
        p.check_invariants();
        let bytes = p.get(location).unwrap();
        assert_eq!(&bytes[..8], &[0xAA; 8]);
        assert_eq!(Handle::from_ne_bytes(bytes[8..].try_into().unwrap()), handle(9));
    }

    /// A write extending past the end of the slot panics.
    #[test]
    #[should_panic(expected = "extends past the end of the slot")]
    fn write_past_slot_end() {
        let mut p = Page::new(0, 0);
        let location = alloc_n(&mut p, 1, idle(0))[0];
        let _ = p.write(location, 14, &[4, 3, 2], HandleMap::Scan(0), idle(0));
    }

    /// `write_all` with more bytes than the slot panics.
    #[test]
    #[should_panic(expected = "bytes exceed slot size")]
    fn write_all_oversized() {
        let mut p = Page::new(0, 0);
        let location = alloc_n(&mut p, 1, idle(0))[0];
        let _ = p.write_all(location, Some(&[0; 17]), HandleMap::Scan(0), idle(0));
    }

    /// A write to an unoccupied slot panics.
    #[test]
    #[should_panic(expected = "location not occupied")]
    fn write_unoccupied() {
        let mut p = Page::new(0, 0);
        let mut location = alloc_n(&mut p, 1, idle(0))[0];
        location.slot += 1;
        let _ = p.write(location, 1, &[4, 3, 2], HandleMap::Scan(0), idle(0));
    }

    /// `write_handle` with an offset outside the slot panics.
    #[test]
    #[should_panic(expected = "lies outside the slot")]
    fn write_handle_offset_out_of_range() {
        let mut p = Page::new(0, 0);
        let location = alloc_n(&mut p, 1, idle(0))[0];
        let _ = p.write_handle(location, 2, Handle(u64::MAX), idle(0));
    }

    /// A write starting partway through a handle-word panics.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "starts inside handle word")]
    fn write_starting_mid_handle() {
        let mut p = Page::new(0, 0);
        let location = alloc_n(&mut p, 1, idle(0))[0];
        let _ = p.write(location, 1, &[4, 3, 2], HandleMap::Scan(1), idle(0));
    }

    /// A write ending partway through a handle-word panics. The write
    /// must start aligned, or the start check fires first.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "stops inside handle word")]
    fn write_ending_mid_handle() {
        let mut p = Page::new(0, 0);
        let location = alloc_n(&mut p, 1, idle(0))[0];
        let _ = p.write(location, 0, &[4, 3, 2], HandleMap::Scan(1), idle(0));
    }

    /// A torn write into a word the layout does not call a handle is
    /// allowed.
    #[test]
    fn torn_write_to_non_handle_is_allowed() {
        let mut p = Page::new(0, 0);
        let location = alloc_n(&mut p, 1, idle(0))[0];
        assert!(!p.write(location, 9, &[4, 3, 2], HandleMap::Scan(1), marking(1)));
        p.check_invariants();
    }
}
mod barrier {
    use super::*;

    /// Only the first barriered write on a page returns true, across
    /// slots and chunks, and `mark_dirty` follows the same rule.
    #[test]
    fn first_write_dirties_page() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, 65, marking(1));
        assert!(p.s_dirty.none());

        assert!(p.write_handle(locs[0], 0, handle(1), marking(1)));
        p.check_invariants();
        assert!(!p.write_handle(locs[1], 0, handle(2), marking(1)));
        assert!(!p.write_handle(locs[64], 1, handle(3), marking(1)));
        assert!(!p.mark_dirty(locs[5], 0));
        p.check_invariants();

        assert_eq!(p.s_dirty, BitMap64::span(0, 2));
        assert_eq!(p.c_dirty[0].count_ones(), 3);
        assert_eq!(p.c_dirty[1].count_ones(), 1);

        // Slot 5's word holds NONE, so the drain skips it
        assert_eq!(drain_sorted(&mut p), raw(&[handle(1), handle(2), handle(3)]));
    }

    /// `NONE`, and kind-0 words with other bits set, never dirty through
    /// any write path.
    #[test]
    fn null_does_not_dirty() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, marking(1))[0];

        assert!(!p.write_handle(l, 0, Handle::NONE, marking(1)));
        assert!(!p.write_handle(l, 1, KIND_ZERO_WORD, marking(1)));

        let mut bytes = [0; 16];
        bytes[8..].copy_from_slice(&KIND_ZERO_WORD.to_ne_bytes());
        assert!(!p.write_all(l, Some(&bytes), HandleMap::Scan(2), marking(1)));
        assert!(!p.write(l, 0, &KIND_ZERO_WORD.to_ne_bytes(), HandleMap::Scan(2), marking(1)));

        p.check_invariants();
        assert!(p.s_dirty.none());
    }

    /// Writes outside marking never dirty, and the same write during
    /// marking does.
    #[test]
    fn idle_write_does_not_dirty() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, marking(1))[0];
        let bytes = handle_bytes(2);

        assert!(!p.write_handle(l, 0, handle(1), idle(0)));
        assert!(!p.write_all(l, Some(&bytes), HandleMap::Scan(2), idle(0)));
        assert!(!p.write(l, 8, &bytes[8..], HandleMap::Scan(2), idle(0)));
        p.check_invariants();
        assert!(p.s_dirty.none());

        assert!(p.write(l, 8, &bytes[8..], HandleMap::Scan(2), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[handle(2)]));
    }

    /// `Scan(n)` dirties exactly the handles inside the overlap of the
    /// scan prefix and the written range.
    #[test]
    fn scan_dirties_range_overlap() {
        let mut p = Page::new(4, 0);
        let l = alloc_n(&mut p, 1, idle(0))[0];
        let bytes = handle_bytes(8);
        assert!(!p.write_all(l, Some(&bytes), HandleMap::Scan(0), idle(0)));

        // Handle-words 2..6 written, prefix is 0..4: only 2 and 3
        assert!(p.write(l, 16, &bytes[16..48], HandleMap::Scan(4), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[handle(3), handle(4)]));

        // Entirely past the prefix: nothing
        assert!(!p.write(l, 32, &bytes[32..], HandleMap::Scan(4), marking(1)));
        p.check_invariants();
        assert!(p.s_dirty.none());

        // Whole slot under a short prefix
        assert!(p.write_all(l, Some(&bytes), HandleMap::Scan(3), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[handle(1), handle(2), handle(3)]));

        // A prefix longer than the slot is clipped to the slot
        assert!(p.write_all(l, Some(&bytes), HandleMap::Scan(100), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&(1..=8).map(handle).collect::<Vec<_>>()));

        // A short write_all only scans the bytes it wrote
        assert!(p.write_all(l, Some(&bytes[..24]), HandleMap::Scan(8), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[handle(1), handle(2), handle(3)]));
    }

    /// `Map` dirties exactly the mapped handles inside a range that
    /// starts and ends mid-word, including across the map's word boundary.
    #[test]
    fn map_dirties_range_overlap() {
        let mut p = Page::new(18, 0);
        let l = alloc_n(&mut p, 1, idle(0))[0];
        let bytes = handle_bytes(128);
        assert!(!p.write_all(l, Some(&bytes), HandleMap::Scan(0), idle(0)));

        // Handle-words 1, 62, 63, 64, 65 and 127 are handles
        let mut w0 = BitMap64::NONE;
        let mut w1 = BitMap64::NONE;
        for hw in [1, 62, 63] { w0.mark(hw); }
        for hw in [0, 1, 63] { w1.mark(hw); }
        let map = [w0, w1];
        let at = |hw: usize| hw * Handle::BYTES;

        // Across the word boundary
        assert!(p.write(l, at(62), &bytes[at(62)..at(66)], HandleMap::Map(&map), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[handle(63), handle(64), handle(65), handle(66)]));

        // A range touching no mapped word
        assert!(!p.write(l, at(2), &bytes[at(2)..at(62)], HandleMap::Map(&map), marking(1)));
        p.check_invariants();
        assert!(p.s_dirty.none());

        // The last word alone
        assert!(p.write(l, at(127), &bytes[at(127)..], HandleMap::Map(&map), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[handle(128)]));

        // The whole slot
        assert!(p.write_all(l, Some(&bytes), HandleMap::Map(&map), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[
            handle(2), handle(63), handle(64), handle(65), handle(66), handle(128),
        ]));

        // A map shorter than the slot reads as no handles past its end
        assert!(p.write_all(l, Some(&bytes), HandleMap::Map(&map[..1]), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[handle(2), handle(63), handle(64)]));
    }

    /// `mark_dirty` reports newly dirty only for the first mark since the
    /// last drain, and repeated marks of one word are idempotent.
    #[test]
    fn mark_dirty_reports_newly_dirty() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, marking(1))[0];
        assert!(!p.write_handle(l, 0, handle(1), idle(0)));

        assert!(p.mark_dirty(l, 0));
        assert!(!p.mark_dirty(l, 0));
        assert!(!p.mark_dirty(l, 1));
        p.check_invariants();
        assert_eq!(p.c_dirty[0].count_ones(), 1);
        assert_eq!(drain_sorted(&mut p), raw(&[handle(1)]));
        assert!(p.mark_dirty(l, 0));
    }

    /// `mark_dirty` with an offset outside the slot panics.
    #[test]
    #[should_panic(expected = "lies outside the slot")]
    fn mark_dirty_offset_out_of_range() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, marking(1))[0];
        let _ = p.mark_dirty(l, 2);
    }

    /// `mark_dirty` on an unoccupied slot panics.
    #[test]
    #[should_panic(expected = "location not occupied")]
    fn mark_dirty_unoccupied() {
        let mut p = Page::new(0, 0);
        let _ = alloc_n(&mut p, 1, marking(1));
        let _ = p.mark_dirty(loc(0, 1), 0);
    }
}

mod drain {
    use super::*;

    /// The drain passes current values: overwritten handles with their
    /// newest value, nulled ones skipped, none reported twice.
    #[test]
    fn reports_current_values() {
        let mut p = Page::new(1, 0);
        let l = alloc_n(&mut p, 1, marking(1))[0];

        assert!(p.write_handle(l, 0, handle(1), marking(1)));
        assert!(!p.write_handle(l, 0, handle(2), marking(1)));
        assert!(!p.write_handle(l, 1, handle(3), marking(1)));
        assert!(!p.write_handle(l, 1, Handle::NONE, marking(1)));
        assert!(!p.write_handle(l, 2, handle(4), marking(1)));
        assert!(!p.write_handle(l, 2, handle(4), marking(1)));
        p.check_invariants();

        assert_eq!(drain_sorted(&mut p), raw(&[handle(2), handle(4)]));
    }

    /// A kind-0 word marked dirty through `mark_dirty` is skipped, but its
    /// marks are still cleared.
    #[test]
    fn skips_kind_zero_word() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, marking(1))[0];
        assert!(!p.write_handle(l, 0, KIND_ZERO_WORD, marking(1)));
        assert!(p.mark_dirty(l, 0));
        p.check_invariants();
        assert_eq!(drain_sorted(&mut p), [] as [u64; 0]);
    }

    /// At 16 B, 32 slots share each dirty word. Several dirty runs in one
    /// word are each reported once, and a word is never reported twice.
    #[test]
    fn small_slots_share_words() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, 64, marking(1));

        let dirty = [0, 1, 5, 31, 32, 40, 63];
        for (i, &slot) in dirty.iter().enumerate() {
            let _ = p.write_handle(locs[slot], (slot % 2) as u32, handle(i as u64 + 1), marking(1));
        }
        p.check_invariants();
        assert_eq!(p.c_dirty[0].count_ones(), 7);

        let expected: Vec<Handle> = (1..=7).map(handle).collect();
        assert_eq!(drain_sorted(&mut p), raw(&expected));
    }

    /// At 48 B a slot is six handle-words, so slot 10 covers handle-words
    /// 60 to 65 and straddles dirty words 0 and 1. Both halves are
    /// drained, along with neighbours on either side.
    #[test]
    fn straddling_slots() {
        let mut p = Page::new(2, 0);
        let locs = alloc_n(&mut p, 12, marking(1));

        // (slot, handle-word in slot, handle): chunk handle-words 59, 63, 64, 66
        let writes = [(9, 5, 1), (10, 3, 2), (10, 4, 3), (11, 0, 4)];
        for (slot, hw, n) in writes {
            let _ = p.write_handle(locs[slot], hw, handle(n), marking(1));
        }
        p.check_invariants();

        let h_dirty = p.h_dirty[0].as_deref().unwrap();
        assert_eq!(h_dirty[0].into_inner(), (1 << 59) | (1 << 63));
        assert_eq!(h_dirty[1].into_inner(), (1 << 0) | (1 << 2));
        assert_eq!(p.c_dirty[0].into_inner(), (1 << 9) | (1 << 10) | (1 << 11));

        assert_eq!(drain_sorted(&mut p), raw(&[handle(1), handle(2), handle(3), handle(4)]));
    }

    /// At 64 KiB a slot spans 128 dirty words. Words at the start, middle
    /// and end of two slots are all drained.
    #[test]
    fn large_slots_span_words() {
        let mut p = Page::new(23, 0);
        let locs = alloc_n(&mut p, 2, marking(1));

        let writes = [(0, 0, 1), (0, 64, 2), (0, 4000, 3), (0, 8191, 4), (1, 0, 5), (1, 8191, 6)];
        for (slot, hw, n) in writes {
            let _ = p.write_handle(locs[slot], hw, handle(n), marking(1));
        }
        p.check_invariants();

        let expected: Vec<Handle> = (1..=6).map(handle).collect();
        assert_eq!(drain_sorted(&mut p), raw(&expected));
    }

    /// Dirty slots in the first, a middle and the last chunk are all
    /// drained, including slot 63 of chunk 63.
    #[test]
    fn last_slot_of_last_chunk() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, PAGE_SLOTS, marking(1));

        let _ = p.write_handle(locs[0], 0, handle(1), marking(1));
        let _ = p.write_handle(locs[2 * 64 + 7], 1, handle(2), marking(1));
        let _ = p.write_handle(locs[PAGE_SLOTS - 1], 1, handle(3), marking(1));
        p.check_invariants();

        let mut expected_chunks = BitMap64::NONE;
        for chunk in [0, 2, 63] { expected_chunks.mark(chunk); }
        assert_eq!(p.s_dirty, expected_chunks);

        assert_eq!(drain_sorted(&mut p), raw(&[handle(1), handle(2), handle(3)]));
    }

    /// After a drain, a second drain reports nothing and the the next
    /// write during marking returns true. Draining a clean page is a
    /// no-op.
    #[test]
    fn leaves_page_clean() {
        let mut p = Page::new(0, 0);
        assert_eq!(drain_sorted(&mut p), [] as [u64; 0]);

        let l = alloc_n(&mut p, 1, marking(1))[0];
        assert!(p.write_handle(l, 0, handle(1), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[handle(1)]));
        assert_eq!(drain_sorted(&mut p), [] as [u64; 0]);

        assert!(p.write_handle(l, 1, handle(2), marking(1)));
        assert_eq!(drain_sorted(&mut p), raw(&[handle(2)]));
    }
}

mod sweep {
    use super::*;

    /// Runs three mark and sweep cycles from `start`. Each cycle keeps
    /// slots 0 and 1, frees 2 and 3, and refills them white. Survivors
    /// must read as unmarked again at the next marking.
    fn survives_three_cycles(start: u64) {
        let mut p = Page::new(0, start);
        let mut epoch = start;
        let locs = alloc_n(&mut p, 4, idle(epoch));

        for cycle in 0..3 {
            epoch += 1;
            for &l in &locs[..2] {
                let (marked, swept) = p.mark_live(l, marking(epoch));
                p.check_invariants();
                assert!(marked, "survivor read as already marked in cycle {cycle}");
                assert_eq!(swept, None);
            }

            let swept = p.sweep_to(idle(epoch).sweep_target());
            p.check_invariants();
            assert_eq!(swept, Some(SweepInfo { freed: 2, became_not_full: false, became_empty: false }));
            assert!(p.is_occupied(locs[0]) && p.is_occupied(locs[1]));
            assert!(!p.is_occupied(locs[2]) && !p.is_occupied(locs[3]));
            assert_eq!(p.marked_live, 0);

            let refill = alloc_n(&mut p, 2, idle(epoch));
            assert_eq!(refill, vec![locs[2], locs[3]]);
        }
    }

    /// Over three cycles from an even index, unmarked objects are freed
    /// and survivors read as unmarked again.
    #[test]
    fn parity_from_even_index() {
        survives_three_cycles(0);
        survives_three_cycles(10);
    }

    /// As `parity_from_even_index`, starting from an odd index.
    #[test]
    fn parity_from_odd_index() {
        survives_three_cycles(1);
        survives_three_cycles(7);
    }

    /// A black allocation survives without `mark_live`, a `mark_live` on
    /// it in the same phase returns false, and it reads as unmarked in
    /// the next marking.
    #[test]
    fn black_alloc_survives_unmarked() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, marking(1))[0];
        assert_eq!(p.marked_live, 1);

        assert_eq!(p.mark_live(l, marking(1)), (false, None));

        let swept = p.sweep_to(1);
        p.check_invariants();
        assert_eq!(swept, Some(SweepInfo::default()));
        assert!(p.is_occupied(l));
        assert_eq!(p.marked_live, 0);

        assert_eq!(p.mark_live(l, marking(2)), (true, None));
        p.check_invariants();
    }

    /// `mark_live` returns true on the first call and false after, and
    /// `marked_live` counts each object once.
    #[test]
    fn mark_live_reports_once() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, idle(0))[0];

        assert_eq!(p.mark_live(l, marking(1)), (true, None));
        assert_eq!(p.mark_live(l, marking(1)), (false, None));
        p.check_invariants();
        assert_eq!(p.marked_live, 1);
    }

    /// `sweep_to` with a gap of zero returns `None`, and neither alloc
    /// nor `mark_live` report a sweep on an up-to-date page.
    #[test]
    fn gap_zero_is_noop() {
        let mut p = Page::new(0, 3);
        assert_eq!(p.sweep_to(3), None);

        let info = p.alloc(None, HandleMap::Scan(0), idle(3)).expect("page has free slots");
        assert_eq!(info.swept, None);

        let (_, swept) = p.mark_live(info.location, marking(4));
        p.check_invariants();
        assert_eq!(swept, None);
    }

    /// `sweep_to` with a gap of one sweeps by marks: only marked objects
    /// survive, and the index advances by one.
    #[test]
    fn gap_one_sweeps() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, 3, idle(0));
        mark_all(&mut p, &locs[1..2], marking(1));

        let swept = p.sweep_to(1);
        p.check_invariants();
        assert_eq!(swept, Some(SweepInfo { freed: 2, became_not_full: false, became_empty: false }));
        assert_eq!(p.sweep_index, 1);
        assert!(!p.is_occupied(locs[0]));
        assert!(p.is_occupied(locs[1]));
        assert!(!p.is_occupied(locs[2]));
    }

    /// `sweep_to` with a gap of two or more resets the page: everything is
    /// freed regardless of marks, chunks stay committed, and the index
    /// jumps straight to the target.
    #[test]
    fn gap_two_resets() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, 70, idle(0));
        mark_all(&mut p, &locs[..1], marking(1));

        let swept = p.sweep_to(2);
        p.check_invariants();
        assert_eq!(swept, Some(SweepInfo { freed: 70, became_not_full: false, became_empty: true }));
        assert!(p.is_empty());
        assert_eq!(p.capacity(), 128);
        assert_eq!(p.sweep_index, 2);
        assert!(!p.is_occupied(locs[0]));

        let info = p.alloc(None, HandleMap::Scan(0), idle(2)).expect("page has free slots");
        p.check_invariants();
        assert_eq!(info.location, loc(0, 0));
        assert_eq!(info.swept, None);

        // A much larger gap also resets
        let mut p = Page::new(0, 0);
        let _ = alloc_n(&mut p, 1, idle(0));
        let swept = p.sweep_to(100);
        p.check_invariants();
        assert_eq!(swept, Some(SweepInfo { freed: 1, became_not_full: false, became_empty: true }));
        assert_eq!(p.sweep_index, 100);
    }

    /// `sweep_to` with a target behind the page panics.
    #[test]
    #[should_panic(expected = "Sweep target behind page's sweep index")]
    fn target_behind_page() {
        let mut p = Page::new(0, 2);
        let _ = p.sweep_to(1);
    }

    /// `became_empty` and `became_not_full` are reported only on real
    /// transitions.
    #[test]
    fn transitions_only_when_real() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, PAGE_SLOTS, idle(0));
        assert!(p.is_full());

        // Everything marked: no frees, no transitions
        mark_all(&mut p, &locs, marking(1));
        assert_eq!(p.sweep_to(1), Some(SweepInfo { freed: 0, became_not_full: false, became_empty: false }));
        assert!(p.is_full());

        // One freed from a full page: not full, not empty
        mark_all(&mut p, &locs[1..], marking(2));
        assert_eq!(p.sweep_to(2), Some(SweepInfo { freed: 1, became_not_full: true, became_empty: false }));
        p.check_invariants();

        // One more freed from a non-full page: no transition
        mark_all(&mut p, &locs[2..], marking(3));
        assert_eq!(p.sweep_to(3), Some(SweepInfo { freed: 1, became_not_full: false, became_empty: false }));
        p.check_invariants();

        // Nothing marked: everything freed, became empty
        assert_eq!(p.sweep_to(4), Some(SweepInfo {
            freed: PAGE_SLOTS as u32 - 2,
            became_not_full: false,
            became_empty: true,
        }));
        p.check_invariants();

        // Sweeping an empty page is no transition
        assert_eq!(p.sweep_to(5), Some(SweepInfo::default()));
        p.check_invariants();
    }

    /// `mark_live` on a page two sweeps behind panics as a stale handle.
    #[test]
    #[should_panic(expected = "untouched by previous marking phase")]
    fn mark_live_gap_two() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, idle(0))[0];
        let _ = p.mark_live(l, marking(3));
    }

    /// `mark_live` on a slot freed by its own catch-up sweep panics.
    #[test]
    #[should_panic(expected = "Marking unoccupied slot")]
    fn mark_live_freed_by_catch_up() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, idle(0))[0];
        // Marking 1 never reached the slot, so catching up to sweep 1 frees it
        let _ = p.mark_live(l, marking(2));
    }

    /// `mark_live` outside the marking phase panics.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Marking outside the marking phase")]
    fn mark_live_outside_marking() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, idle(0))[0];
        let _ = p.mark_live(l, idle(0));
    }

    /// Dirty marks on survivors are kept through a catch-up sweep during
    /// marking, and the drain still reports them.
    #[test]
    fn catch_up_keeps_survivor_dirty() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, 2, idle(0));

        // Marking 1 reaches only slot 0; the page stays at index 0
        mark_all(&mut p, &locs[..1], marking(1));

        // Marking 2 begins before the page is swept, and the mutator
        // writes into the survivor through the barrier
        assert!(p.write_handle(locs[0], 1, handle(7), marking(2)));
        p.check_invariants();

        // Tracing the survivor catches the page up, freeing slot 1
        let (marked, swept) = p.mark_live(locs[0], marking(2));
        p.check_invariants();
        assert!(marked);
        assert_eq!(swept, Some(SweepInfo { freed: 1, became_not_full: false, became_empty: false }));
        assert!(p.s_dirty.any());

        assert_eq!(drain_sorted(&mut p), raw(&[handle(7)]));
    }

    /// A sweep that would free a dirty slot panics.
    #[test]
    #[should_panic(expected = "no dirty marks on dead slot")]
    fn dirty_freed_slot() {
        let mut p = Page::new(0, 0);
        let locs = alloc_n(&mut p, 2, idle(0));
        mark_all(&mut p, &locs[..1], marking(1));

        // A write into the dead slot 1 can only come from a stale handle
        let _ = p.write_handle(locs[1], 0, handle(1), marking(1));
        let _ = p.sweep_to(1);
    }

    /// A reset of a page with any dirty mark panics.
    #[test]
    #[should_panic(expected = "Dirty mark in dead page")]
    fn dirty_reset_page() {
        let mut p = Page::new(0, 0);
        let l = alloc_n(&mut p, 1, idle(0))[0];
        let _ = p.write_handle(l, 0, handle(1), marking(1));
        let _ = p.sweep_to(2);
    }
}

mod model {

    use super::*;

    /// xorshift64, enough randomness for exercising a bitmap allocator.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn chance(&mut self, percent: u64) -> bool {
            self.below(100) < percent
        }

        fn pick<T: Copy>(&mut self, items: &[T]) -> Option<T> {
            if items.is_empty() { None } else { Some(items[self.below(items.len() as u64) as usize]) }
        }
    }

    type Key = (u32, u32);
    type Word = (u32, u32, u32);

    /// A reference page: occupancy, the sweep each object is known to
    /// survive, slot contents by handle-word, and the dirty set.
    ///
    /// An object survives sweep `s` exactly when `survives >= s`. A white
    /// allocation at idle `e` and a black allocation or mark at marking
    /// `e` all give `survives = e`, so the model never needs to know the
    /// page's mark parity.
    ///
    struct Model {
        slot_hw: u32,
        sweep_index: u64,
        objects: BTreeMap<Key, u64>,
        words: BTreeMap<Word, u64>,
        dirty: BTreeSet<Word>,
    }

    impl Model {
        fn new(slot_hw: u32, sweep_index: u64) -> Self {
            Self {
                slot_hw, sweep_index,
                objects: BTreeMap::new(), words: BTreeMap::new(), dirty: BTreeSet::new(),
            }
        }

        fn lowest_free(&self) -> Option<Key> {
            (0..PAGE_SLOTS as u32)
                .map(|i| (i >> PAGE_ENTRY_BITS, i & PAGE_ENTRY_MASK))
                .find(|key| !self.objects.contains_key(key))
        }

        fn is_full(&self) -> bool {
            self.objects.len() == PAGE_SLOTS
        }

        /// Objects the mutator may legally touch under `target`: those
        /// that survived the last completed marking or were allocated
        /// since. Anything else is dead, and touching it is a stale handle.
        fn live(&self, target: u64) -> Vec<Key> {
            self.objects.iter().filter(|(_, s)| **s >= target).map(|(&k, _)| k).collect()
        }

        /// Mirrors `Page::sweep_to`, returning what the page should report.
        fn sweep_to(&mut self, target: u64) -> Option<SweepInfo> {
            let gap = target - self.sweep_index;
            if gap == 0 { return None; }

            let was_full  = self.is_full();
            let was_empty = self.objects.is_empty();

            let dead: Vec<Key> = self.objects.iter()
                .filter(|(_, s)| **s < target)
                .map(|(&k, _)| k)
                .collect();
            for &(chunk, slot) in &dead {
                self.objects.remove(&(chunk, slot));
                for hw in 0..self.slot_hw {
                    assert!(!self.dirty.contains(&(chunk, slot, hw)), "model dirtied a dead object");
                    self.words.remove(&(chunk, slot, hw));
                }
            }
            if gap >= 2 {
                assert!(self.objects.is_empty(), "a page two sweeps behind must hold only dead objects");
            }
            self.sweep_index = target;

            Some(SweepInfo {
                freed: dead.len() as u32,
                became_not_full: was_full && !self.is_full(),
                became_empty: !was_empty && self.objects.is_empty(),
            })
        }

        fn slot_bytes(&self, (chunk, slot): Key) -> Vec<u8> {
            (0..self.slot_hw)
                .flat_map(|hw| self.words.get(&(chunk, slot, hw)).copied().unwrap_or(0).to_ne_bytes())
                .collect()
        }

        /// Records a write of `value` at `hw`, returning whether the page
        /// should report newly dirty.
        fn write(&mut self, (chunk, slot): Key, hw: u32, value: u64, gc: GcPhaseInfo) -> bool {
            let was_clean = self.dirty.is_empty();
            self.words.insert((chunk, slot, hw), value);
            if gc.marking() && value != 0 {
                self.dirty.insert((chunk, slot, hw));
                was_clean
            } else {
                false
            }
        }

        fn expected_drain(&self) -> Vec<u64> {
            let mut seen: Vec<u64> = self.dirty.iter()
                .map(|w| self.words.get(w).copied().unwrap_or(0))
                .filter(|&v| v != 0)
                .collect();
            seen.sort_unstable();
            seen
        }

        /// Compares the page against the model after an op.
        fn check(&self, p: &Page, rng: &mut Rng) {
            p.check_invariants();
            assert_eq!(p.len(), self.objects.len());
            assert_eq!(p.is_full(), self.is_full());
            assert_eq!(p.is_empty(), self.objects.is_empty());
            assert_eq!(p.sweep_index, self.sweep_index);
            assert_eq!(p.s_dirty.any(), !self.dirty.is_empty());

            let keys: Vec<Key> = self.objects.keys().copied().collect();
            for &(chunk, slot) in &keys {
                assert!(p.is_occupied(loc(chunk, slot)));
            }
            for _ in 0..4 {
                if let Some(key) = rng.pick(&keys) {
                    assert_eq!(p.get(loc(key.0, key.1)).unwrap(), &self.slot_bytes(key)[..]);
                }
            }
        }
    }

    /// Random slot contents: each handle-word null or a fresh handle.
    fn random_words(rng: &mut Rng, slot_hw: u32, next: &mut u64) -> Vec<u64> {
        (0..slot_hw).map(|_| if rng.chance(40) { 0 } else { *next += 1; handle(*next).0 }).collect()
    }

    fn to_bytes(words: &[u64]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_ne_bytes()).collect()
    }

    /// Drives a page and the model through `ops` random operations: white
    /// and black allocation, barriered and unbarriered writes, marking,
    /// explicit sweeps, and phase changes with a drain before each atomic
    /// step. The page is only swept when the model is, so lagging pages,
    /// catch-up sweeps and resets all occur naturally.
    ///
    fn run(seed: u64, slot_size: u32, ops: usize) {
        let slot_size_class = class_for(slot_size);
        let slot_hw = slot_size / Handle::BYTES as u32;
        let mut rng = Rng(seed);
        let mut p = Page::new(slot_size_class, 0);
        let mut m = Model::new(slot_hw, 0);
        let mut epoch = 0;
        let mut marking_now = false;
        let mut next_handle = 0;

        for op in 0..ops {
            let gc = GcPhaseInfo::new(epoch, marking_now);
            let target = gc.sweep_target();

            match rng.below(100) {
                0..35 => {
                    let swept = m.sweep_to(target);
                    let expected = m.lowest_free();
                    let words = random_words(&mut rng, slot_hw, &mut next_handle);
                    let bytes = to_bytes(&words);
                    let init = if rng.chance(60) { Some(&bytes[..]) } else { None };

                    let result = p.alloc(init, HandleMap::Scan(slot_hw), gc);
                    match expected {
                        None => assert_eq!(result.err(), Some(AllocError::PageFull), "op {op}"),
                        Some(key) => {
                            let info = result.expect("model says a slot is free");
                            assert_eq!(info.location, loc(key.0, key.1), "op {op}");
                            assert_eq!(info.swept, swept, "op {op}");

                            m.objects.insert(key, epoch);
                            let mut dirtied = false;
                            for hw in 0..slot_hw {
                                let value = init.map_or(0, |_| words[hw as usize]);
                                dirtied |= m.write(key, hw, value, gc);
                            }
                            assert_eq!(info.was_dirtied, dirtied, "op {op}");
                            assert_eq!(info.became_full, m.is_full(), "op {op}");
                        }
                    }
                }
                35..60 => {
                    let Some(key) = rng.pick(&m.live(target)) else { continue };
                    let location = loc(key.0, key.1);
                    match rng.below(3) {
                        0 => {
                            let hw = rng.below(u64::from(slot_hw)) as u32;
                            let value = if rng.chance(30) { 0 } else { next_handle += 1; handle(next_handle).0 };
                            let reported = p.write_handle(location, hw, Handle(value), gc);
                            assert_eq!(reported, m.write(key, hw, value, gc), "op {op}");
                        }
                        1 => {
                            let words = random_words(&mut rng, slot_hw, &mut next_handle);
                            let reported = p.write_all(
                                location, Some(&to_bytes(&words)), HandleMap::Scan(slot_hw), gc,
                            );
                            let mut expected = false;
                            for (hw, &value) in words.iter().enumerate() {
                                expected |= m.write(key, hw as u32, value, gc);
                            }
                            assert_eq!(reported, expected, "op {op}");
                        }
                        _ => {
                            let reported = p.write_all(location, None, HandleMap::Scan(slot_hw), gc);
                            for hw in 0..slot_hw { let _ = m.write(key, hw, 0, gc); }
                            assert!(!reported, "op {op}");
                        }
                    }
                    assert_eq!(p.get(location).unwrap(), &m.slot_bytes(key)[..], "op {op}");
                }
                60..78 if marking_now => {
                    let Some(key) = rng.pick(&m.live(target)) else { continue };
                    let swept = m.sweep_to(target);
                    let (marked, reported) = p.mark_live(loc(key.0, key.1), gc);
                    assert_eq!(reported, swept, "op {op}");
                    let survives = m.objects.get_mut(&key).unwrap();
                    assert_eq!(marked, *survives != epoch, "op {op}");
                    *survives = epoch;
                }
                60..88 => {
                    let swept = m.sweep_to(target);
                    assert_eq!(p.sweep_to(target), swept, "op {op}");
                }
                _ => {
                    if marking_now {
                        // The atomic step: every dirty page is drained first
                        assert_eq!(drain_sorted(&mut p), m.expected_drain(), "op {op}");
                        m.dirty.clear();
                        marking_now = false;
                    } else {
                        epoch += 1;
                        marking_now = true;
                    }
                }
            }

            m.check(&p, &mut rng);
        }

        // Finish cleanly so the final drain is checked too
        if marking_now {
            assert_eq!(drain_sorted(&mut p), m.expected_drain());
        }
    }

    /// Random op sequences match the model at the smallest class, where
    /// 32 slots share each dirty word.
    #[test]
    fn random_ops_match_model_16() {
        for seed in 1..=3 {
            run(seed, 16, 4000);
        }
    }

    /// Random op sequences match the model at 48 B, where slots straddle
    /// dirty words.
    #[test]
    fn random_ops_match_model_48() {
        for seed in 11..=13 {
            run(seed, 48, 4000);
        }
    }

    /// A longer run so the page has a chance to fill and report `PageFull`
    /// against the model.
    #[test]
    fn random_ops_long_run() {
        run(99, 16, 16000);
    }
}
