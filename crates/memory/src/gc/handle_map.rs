// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

use crate::bitmap::BitMap64;

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
                let word = (handle_hw >> BitMap64::INDEX_BITS) as usize;
                word < map.len() && map[word].get(handle_hw & BitMap64::INDEX_MASK)
            },
        }
    }

}
