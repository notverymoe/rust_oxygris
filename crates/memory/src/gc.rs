// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

//! The collector's side of the page interface: the phase snapshot that
//! decides how pages allocate, sweep and apply the write barrier, and the
//! handle layouts the barrier uses to find handles in a slot.
//!
//! Pages never store either. Both are passed in on each call that needs
//! them, so the collector stays the single source of truth for the phase
//! and the type system for layouts. The collector itself (worklists,
//! pacing and the atomic step) is not yet written.

mod handle_map;
pub use handle_map::HandleMap;

mod phase_info;
pub use phase_info::GcPhaseInfo;
