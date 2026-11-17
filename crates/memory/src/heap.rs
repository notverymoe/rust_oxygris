// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

//! The size-class heap: fixed-size slots grouped into pages, and the
//! locations that address them.
//!
//! - [`Page`]: up to 64 chunks of 64 slots of one size class, with its
//!   own occupancy, mark and dirty state, and its own sweep index.
//! - [`HeapLoc`]: a slot's address in the whole heap, as stored in a
//!   handle. [`PageLoc`] is the part of it a page understands.
//! - [`util`]: the size-class table mapping classes to slot sizes.
//! - [`HeapAllocError`]: why an allocation failed.
//!
//! A page doesn't know its own id. Whatever owns the pages joins a
//! page's id with the [`PageLoc`] it returns to form a [`HeapLoc`], and
//! splits a [`HeapLoc`] back apart to find the page and the slot in it.

mod directory;
mod error;
mod location;
mod page;
pub mod util;

pub use error::HeapAllocError;
pub use location::{HeapLoc, PageLoc};
pub use page::{Page, PageAllocInfo, PageSweepInfo};