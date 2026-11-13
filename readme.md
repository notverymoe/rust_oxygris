# Oxygris 

A statically typed, sandboxed scripting language and virtual machine for games, written in Rust.


## Status

This is a hobby project, not intended for production use. I'm utilizing it
to learn about virtual machine design and programming language design. One
day it may become production ready, but that time is not soon or definite.

## Why

When looking at scripting languages available, none of them quite met my
requirements:

- Static Typing 
    - To catch errors before a script runs
    - The performance benefits of static lookups over dynamic tables
- Value Types
    - This allows nested structures to lay flat in memory reducing GC pressure
    - Further, value types can also be stored in the stack frames, reducing the
      cost of temporaries
- Embedding-first
    - Sandboxing, Fuel Budget, Memory Limits to allow running untrusted
      code more safely
    - Minimal compiled size without large compilation toolchain
    - Minimal runtime size
- Desktop, Android & WASM targets

## What

- Syntax that draws from Rust and TypeScript, light on shifted characters
- Record types with explicit layouts for interop (c, std140, std430)
- First-class vector types, so game math costs fewer interpreter dispatches
- Second-class references that cannot be stored
- Coroutine-centric execution model, where async is opt-out
- A compact register-based VM that compiles a typed IR produced by a separate frontend
- An incremental garbage collector that runs over a size-segregated paged heap

## Pros

- Type and layout errors manifest at compile time
- Script data can be handed to the GPU or host without conversion
- Execution can be budgeted and paused at well-defined points
- A compact runtime with few dependencies
    - The VM is also no_std + alloc

## Cons

- New language without an ecosystem of tooling and libraries
- Interpreter-based, with no plans for JIT compilation
- Garbage-collection pauses are bounded but not absent
- Everything will be in flux for a long time

## AI Use Disclosure (Large Language Models)

Parts of this repository were written by or with the assistance of a
large language model. In the interest of transparency this includes:
- Function and module-level comments
- Unit tests
- Code review
- High-level concept review

It should be noted that by default the unit tests have not been closely
scrutinized. Human-reviewed or human-written unit tests will always be
marked as such. If you do not want to run un-reviewed LLM output, please
AVOID running `cargo test`.

This was done to fill the gaps in tasks I rarely complete in unserious
hobby projects. It's to make the most of the limited development time I
have, and is the reason I don't recommend the use of this project in a
production use-case. This project should only be considered experimental
or educational. At some point in the future it may be rewritten without
AI involvement.

The important thing to me is that the operating runtime code has been
designed and written by me, but I recognize that this usage of AI is
an ethical compromise on my part.

## License

Licensed under either of the MIT license or the Apache License,
Version 2.0, at your option.