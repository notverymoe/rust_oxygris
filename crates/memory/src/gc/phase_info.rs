// Oxygris | Copyright 2026 Natalie Baker | MIT / Apache License v2.0 //

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
    /// If `mark_epoch` is `0` when `marking` is `true`.
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