//! WebAssembly exports: a plain C interface over [`Engine`], so the page needs no generated
//! bindings. Strings cross the boundary as UTF-8 bytes in the module's memory:
//!
//! 1. `sim_alloc(len)` → pointer; the caller writes the bytes there.
//! 2. `sim_load(model, model_len, params, params_len)` once, then `sim_simulate(request, len)`
//!    as often as needed. Each takes ownership of its buffers and returns 1 on success.
//! 3. `sim_output_ptr()` / `sim_output_len()`: the JSON result, or the error message.

use crate::engine::Engine;
use std::cell::RefCell;

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    static OUTPUT: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Take back a buffer handed out by `sim_alloc` and filled by the caller.
unsafe fn take(ptr: *mut u8, len: usize) -> Result<String, String> {
    // SAFETY: `ptr` came from `sim_alloc(len)`, which leaked a `Vec` of exactly this capacity.
    let bytes = unsafe { Vec::from_raw_parts(ptr, len, len) };
    String::from_utf8(bytes).map_err(|e| e.to_string())
}

fn finish(result: Result<String, String>) -> u32 {
    let ok = result.is_ok();
    OUTPUT.with(|o| *o.borrow_mut() = result.unwrap_or_else(|e| e));
    u32::from(ok)
}

#[no_mangle]
pub extern "C" fn sim_alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// # Safety
/// Both buffers must come from `sim_alloc` with these lengths and be fully written.
#[no_mangle]
pub unsafe extern "C" fn sim_load(model: *mut u8, model_len: usize, params: *mut u8, params_len: usize) -> u32 {
    let loaded = unsafe { take(model, model_len).and_then(|m| Ok((m, take(params, params_len)?))) }
        .and_then(|(m, p)| Engine::new(&m, &p));
    finish(loaded.map(|engine| {
        ENGINE.with(|e| *e.borrow_mut() = Some(engine));
        String::new()
    }))
}

/// # Safety
/// The buffer must come from `sim_alloc` with this length and be fully written.
#[no_mangle]
pub unsafe extern "C" fn sim_simulate(request: *mut u8, len: usize) -> u32 {
    let request = unsafe { take(request, len) };
    finish(request.and_then(|r| ENGINE.with(|e| match e.borrow().as_ref() {
        Some(engine) => engine.simulate(&r),
        None => Err("call sim_load first".to_string()),
    })))
}

#[no_mangle]
pub extern "C" fn sim_output_ptr() -> *const u8 {
    OUTPUT.with(|o| o.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn sim_output_len() -> usize {
    OUTPUT.with(|o| o.borrow().len())
}
