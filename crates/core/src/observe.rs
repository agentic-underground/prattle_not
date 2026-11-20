//! Where an attention layer hands over its probabilities `[1, heads, queries, keys]`. Nothing listens
//! unless a tool installs a listener on this thread; inference is unchanged either way.

use candle_core::{Result, Tensor};
use std::cell::RefCell;

type Listener = Box<dyn Fn(&Tensor) -> Result<()>>;

thread_local! {
    static LISTENER: RefCell<Option<Listener>> = const { RefCell::new(None) };
}

/// Hand every attention layer's probabilities on this thread to `f`, replacing any listener before it.
pub fn listen(f: impl Fn(&Tensor) -> Result<()> + 'static) {
    LISTENER.with(|l| *l.borrow_mut() = Some(Box::new(f)));
}

/// Stop listening on this thread.
pub fn stop() {
    LISTENER.with(|l| *l.borrow_mut() = None);
}

/// Whether a listener is installed on this thread — an attention layer then keeps its whole matrix.
pub fn listening() -> bool {
    LISTENER.with(|l| l.borrow().is_some())
}

/// Called by an attention layer with its probabilities.
pub fn attention(p: &Tensor) -> Result<()> {
    LISTENER.with(|l| match l.borrow().as_ref() {
        Some(f) => f(p),
        None => Ok(()),
    })
}
